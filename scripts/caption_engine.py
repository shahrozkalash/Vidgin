#!/usr/bin/env python3
"""Vidgin Caption Engine. Offline client for locally hosted vision-language models.
Python 3.11; requires numpy and requests; optional cv2 for MP4 and optical flow.
"""
import concurrent.futures, argparse, base64, hashlib, io, json, os, re, shutil, signal, statistics, sys, tempfile, time, urllib.parse, random, collections
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from collections import Counter
import requests
try:
    import cv2
except ImportError:
    cv2 = None

STOP = False
VAGUE_WORDS={'various','some','stuff','things','nice','beautiful','amazing','interesting','several'}
CONFIDENCE_THRESHOLD={'motion':0.35,'zoom':0.45,'depth':0.30}

FIELDS = ('subject','action','setting','camera_motion','lighting')

def stop_handler(sig, frame):
    global STOP
    STOP = True


def canonical_hash(arr, depth=None):
    h = hashlib.sha256()
    h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode()); h.update(arr.tobytes())
    if depth is not None:
        h.update(str(depth.shape).encode()); h.update(str(depth.dtype).encode()); h.update(depth.tobytes())
    return h.hexdigest()


def load_clip(path):
    if path.suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False, mmap_mode='r')
        if a.ndim != 4 or a.shape[-1] != 3 or a.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB [T,H,W,3] array')
        depth_path = path.with_name(path.stem + '_depth.npy')
        depth = np.load(depth_path, allow_pickle=False, mmap_mode='r') if depth_path.exists() else None
        if depth is not None and (depth.ndim < 3 or depth.shape[0] != a.shape[0] or depth.shape[1:3] != a.shape[1:3]):
            raise ValueError('Depth shape does not match RGB frames')
        return a, depth
    if cv2 is None:
        raise RuntimeError('MP4 support requires optional opencv-python')
    cap = cv2.VideoCapture(str(path)); frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally: cap.release()
    if not frames: raise ValueError('No decoded video frames')
    return np.stack(frames), None


def sample_indices(frames, n=8):
    t = len(frames)
    if not t: raise ValueError('Empty clip')
    uniform = np.linspace(0,t-1,min(n,t),dtype=int).tolist()
    # Every adjacent frame pair is measured on a small luminance proxy.
    proxy = frames[:,::max(1,frames.shape[1]//64),::max(1,frames.shape[2]//64)].astype(np.float32)
    grey = proxy.mean(axis=-1)
    diffs = np.abs(np.diff(grey,axis=0)).mean(axis=(1,2)) if t>1 else np.array([])
    peaks = (np.argsort(diffs)[::-1][:min(3,len(diffs))]+1).tolist()
    return sorted(set(uniform+peaks)), diffs


CAMERA = {'static','pan_left','pan_right','tilt_up','tilt_down','zoom_in','zoom_out','moving','unknown'}
LIGHT = {'bright','even','dim','low_light','high_contrast','unknown'}
DISTANCE = {'near','mid','far','unknown'}

NORMALIZATION_JSON = r'''{"automobile":"car","automobiles":"cars","folks":"people","couch":"sofa","couches":"sofas","kid":"child","kids":"children","vehicle":"car","bicycle rider":"cyclist","pedestrians":"people"}'''
NORMALIZATION = json.loads(NORMALIZATION_JSON)
NORMALIZATION_COUNT = 0

def normalize(s, count=False):
    global NORMALIZATION_COUNT
    s=str(s)
    for old,new in sorted(NORMALIZATION.items(),key=lambda kv:-len(kv[0])):
        s,n=re.subn(r'(?<!\w)'+re.escape(old)+r'(?!\w)',new,s,flags=re.I)
        if count: NORMALIZATION_COUNT+=n
    return s

def shard_for(filename, total):
    return int(hashlib.sha256(Path(filename).name.encode('utf-8')).hexdigest(),16)%total+1

def validate_local(url):
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='http' or parsed.hostname not in ('localhost','127.0.0.1','::1') or parsed.username or parsed.password:
        raise ValueError('Only HTTP loopback endpoints without credentials allowed')

def classify_error(e):
    if isinstance(e,requests.exceptions.Timeout):return 'timeout'
    if isinstance(e,requests.exceptions.RequestException):return 'http'
    if isinstance(e,(json.JSONDecodeError,KeyError,TypeError)) or 'JSON' in str(e):return 'bad_json'
    return 'other'

def robust_describe(backend,frames,facts):
    last=None
    for attempt in range(3):
        try:
            result=backend.describe(frames,facts)
            if not isinstance(result,dict):raise ValueError('bad JSON object')
            return result
        except (json.JSONDecodeError,KeyError,TypeError,ValueError) as ex:
            last=ex
            if attempt==0:
                try:
                    result=backend.describe(frames,dict(facts,repair_request='Return valid JSON only; no markdown or explanation.'))
                    if isinstance(result,dict):return result
                except Exception as repair_ex:last=repair_ex
        except Exception as ex:last=ex
        if attempt<2:time.sleep(.15*(2**attempt))
    raise last

def scale_estimate(a,b):
    # Search feature-spread scale around the frame center using low-resolution image registration.
    # A zoom-in increases apparent feature scale. Mean-absolute-error alignment is evaluated
    # on a central crop, reducing black-border effects.
    h,w=a.shape
    size=min(h,w,96)
    ys=np.linspace(h*.20,h*.80,32).astype(int); xs=np.linspace(w*.20,w*.80,32).astype(int)
    base=a[np.ix_(ys,xs)].astype(np.float32)
    candidates=np.linspace(.88,1.16,15)
    scores=[]
    for scale in candidates:
        yy=np.clip(np.rint((ys-h/2)/scale+h/2).astype(int),0,h-1)
        xx=np.clip(np.rint((xs-w/2)/scale+w/2).astype(int),0,w-1)
        # Evaluate scale in the second frame against the first frame.
        pred=b[np.ix_(yy,xx)].astype(np.float32)
        scores.append(float(np.mean(np.abs(base-pred))))
    best=int(np.argmin(scores))
    return float(candidates[best]), float(np.median(scores)-scores[best])

def phase_shift(a,b):
    # Phase correlation estimates global translation (dx,dy) from a to b.
    a=a.astype(np.float32); b=b.astype(np.float32)
    h,w=a.shape; window=np.outer(np.hanning(h),np.hanning(w)).astype(np.float32)
    fa=np.fft.fft2((a-a.mean())*window); fb=np.fft.fft2((b-b.mean())*window)
    cross=fa*np.conj(fb); cross/=np.maximum(np.abs(cross),1e-8)
    corr=np.abs(np.fft.ifft2(cross)); peak=np.unravel_index(np.argmax(corr),corr.shape)
    dy,dx=(int(peak[0]),int(peak[1])); dy=dy if dy<=h//2 else dy-h; dx=dx if dx<=w//2 else dx-w
    confidence=float(corr[peak]/max(np.mean(corr),1e-8))
    return -dx,-dy,confidence

def motion_analysis(frames,indices):
    shifts=[]; residuals=[]; confidences=[]; scales=[]; masks=[]; method='phase_correlation_numpy'
    for i,j in zip(indices[:-1],indices[1:]):
        a=frames[i].astype(np.float32).mean(axis=2); b=frames[j].astype(np.float32).mean(axis=2)
        if float(np.mean(np.abs(a-b)))>100 or min(float(a.std()),float(b.std()))<3:continue
        scale,score=scale_estimate(a,b)
        if score>1.5:scales.append(scale)
        if cv2 is not None:
            flow=cv2.calcOpticalFlowFarneback(a.astype(np.uint8),b.astype(np.uint8),None,.5,3,21,3,5,1.2,0)
            dx,dy=np.median(flow.reshape(-1,2),axis=0)
            residual=np.linalg.norm(flow-np.array([dx,dy]),axis=2)
            method='farneback_median_flow'
            confidence=float(np.mean(residual<max(2.,np.hypot(dx,dy)*2)))
            mask=residual>max(1.2,float(np.percentile(residual,75)))
            residual_strength=float(np.median(residual))
        else:
            dx,dy,confidence=phase_shift(a[::2,::2],b[::2,::2]);dx*=2;dy*=2
            aligned=np.roll(a,(int(round(dy)),int(round(dx))),(0,1))
            residual=np.abs(b-aligned)
            threshold=max(12.,float(np.percentile(residual,85)))
            mask=residual>threshold
            residual_strength=float(np.mean(residual)/255)
        masks.append(mask);shifts.append((float(dx)/(j-i),float(dy)/(j-i)))
        residuals.append(residual_strength);confidences.append(float(confidence))
    if not shifts:
        static=bool(len(indices)>1 and np.max(np.abs(np.diff(frames[indices].astype(np.int16),axis=0)))<3)
        return {'camera_motion':'static' if static else 'unknown','subject_motion':'unknown','motion_method':method,'motion_confidence':0.,'_subject_mask':None}
    dx,dy=np.median(shifts,axis=0);speed=float(np.hypot(dx,dy))
    scale_med=float(np.median(scales)) if scales else 1.
    # Scale candidate requires >=2 consistent pairs; single accidental scale does not trigger zoom.
    zoom=len(scales)>=2 and abs(scale_med-1)>=.035 and np.mean(np.sign(np.array(scales)-1)==np.sign(scale_med-1))>=.7
    if zoom:label='zoom_in' if scale_med<1 else 'zoom_out'
    elif speed<.12:label='static'
    elif abs(dx)>abs(dy)*1.35:label='pan_left' if dx>0 else 'pan_right'
    elif abs(dy)>abs(dx)*1.35:label='tilt_up' if dy>0 else 'tilt_down'
    else:label='moving'
    # Select the most substantial residual-motion region, avoiding diffuse noise.
    mask=None
    if masks:
        votes=np.mean(np.stack(masks),axis=0)
        mask=votes>=.30
        if mask.mean()<.003 or mask.mean()>.55:mask=None
        else:
            # Expand residual-motion edges into the moving object's interior.
            expanded=mask.copy()
            for dy in (-6,-3,0,3,6):
                for dx in (-6,-3,0,3,6):expanded|=np.roll(mask,(dy,dx),(0,1))
            mask=expanded if expanded.mean()<.60 else mask
    return {'camera_motion':label,'subject_motion':('low' if np.median(residuals)<(.7 if cv2 is not None else .08) else 'moderate_or_high')+' residual motion',
            'global_dx_per_frame':round(float(dx),3),'global_dy_per_frame':round(float(dy),3),
            'scale_ratio':round(scale_med,4),'motion_confidence':round(float(np.median(confidences)),3),
            'motion_method':method,'_subject_mask':mask}

def analyze(frames, depth, n=8):
    idx,diffs=sample_indices(frames,n)
    selected=np.asarray(frames[idx],dtype=np.uint8)
    lum=selected.astype(np.float32) @ np.array([.2126,.7152,.0722],dtype=np.float32)
    mean=float(np.mean(lum)); contrast=float(np.std(lum)); dark=float(np.mean(lum<35))
    lighting=('low_light' if mean<45 or dark>.75 else 'high_contrast' if contrast>75 else
              'bright' if mean>=180 else 'dim' if mean<95 else 'even')
    med=float(np.median(diffs)) if len(diffs) else 0.; peak=float(np.max(diffs)) if len(diffs) else 0.
    facts={'sample_indices':idx,'median_frame_difference':round(med,2),'peak_frame_difference':round(peak,2),
           'scene_cut':bool(peak>65 and peak>3*max(med,1)), 'near_static':bool(med<2.5),
           'near_black':bool(dark>.8),'lighting':lighting,
           'luminance_mean':round(mean,2),'luminance_std':round(contrast,2),'dark_share':round(dark,3)}
    motion=motion_analysis(frames,idx)
    mask=motion.pop('_subject_mask',None)
    facts.update(motion)
    facts['subject_distance']='unknown';facts['subject_placement']='unknown'
    if depth is not None:
        d=np.asarray(depth[idx],dtype=np.float32)
        h,w=d.shape[1:3]
        if mask is None or mask.shape!=(h,w):
            mask=np.zeros((h,w),dtype=bool);mask[h//4:3*h//4,w//4:3*w//4]=True
            region='center_fallback'
        else:region='residual_motion'
        inside=d[:,mask];outside=d[:,~mask]
        inside=inside[np.isfinite(inside)];outside=outside[np.isfinite(outside)]
        if inside.size and outside.size:
            c=float(np.percentile(inside,90) if region=='residual_motion' else np.median(inside));e=float(np.median(outside))
            valid=d[np.isfinite(d)];spread=float(np.percentile(valid,90)-np.percentile(valid,10))
            rel=(c-e)/max(spread,abs(c-e),1e-6)
            facts['subject_distance']='near' if rel>.2 else 'far' if rel<-.2 else 'mid'
            facts['subject_placement']='foreground' if rel>.2 else 'background' if rel<-.2 else 'undetermined'
            yy,xx=np.where(mask)
            side='left' if xx.mean()<w*.4 else 'right' if xx.mean()>w*.6 else 'center'
            facts['subject_side']=side
            facts['depth']={'subject_median':round(c,4),'outside_median':round(e,4),'relative_contrast':round(rel,3),'polarity':'larger_is_nearer','region':region,'side':side}
    return selected,facts

def measurement_confidence(frames, depth, facts):
    """Conservative heuristic: texture variance, pair agreement, depth contrast and mask coverage.
    Scores are heuristics, not calibrated probabilities.
    """
    indices=facts.get('sample_indices',list(range(min(len(frames),8))))
    gray=np.stack([np.asarray(frames[i],dtype=np.float32).mean(axis=2)[::4,::4] for i in indices])
    texture=float(np.median(np.std(gray,axis=(1,2))))
    texture_score=float(np.clip((texture-5)/30,0,1))
    diffs=np.abs(np.diff(gray,axis=0)).mean(axis=(1,2)) if len(gray)>1 else np.zeros(1)
    # Pure static scenes with sufficient texture are confidently static.
    motion=float(np.clip(texture_score*(.8 if np.std(diffs)<20 else .6),0,1))
    zoom=float(np.clip(motion*(.85 if facts.get('camera_motion') in ('zoom_in','zoom_out') else .5),0,1))
    if depth is None: dc=0.
    else:
        d=np.asarray(depth[indices[len(indices)//2]],dtype=np.float32)
        finite=d[np.isfinite(d)]
        contrast=float(np.percentile(finite,90)-np.percentile(finite,10)) if finite.size else 0
        dc=float(np.clip(contrast/(abs(float(np.median(finite)))+1e-5),0,1)) if finite.size else 0.
    return {'motion':round(motion,3),'zoom':round(zoom,3),'depth':round(dc,3)}


def confidence_adjust(frames,depth,facts):
    facts=dict(facts)
    confidence=measurement_confidence(frames,depth,facts)
    facts['vidgin_measurement_confidence']=confidence
    if confidence['motion']<CONFIDENCE_THRESHOLD['motion']:
        facts['camera_motion']='unknown'
        facts['motion_reliability']='unreliable'
    if facts['camera_motion'] in ('zoom_in','zoom_out') and confidence['zoom']<CONFIDENCE_THRESHOLD['zoom']:
        facts['camera_motion']='unknown'
    if depth is not None and confidence['depth']<CONFIDENCE_THRESHOLD['depth']:
        facts['subject_distance']='unknown';facts['subject_placement']='undetermined'
        facts['depth_reliability']='unreliable'
    return facts


def skeptic_prompt(caption,positions):
    return ('INDEPENDENT ADVERSARIAL VISUAL CHECK. You have NOT seen the draft observations. '
            'Break this caption into individual claims. For each claim answer SUPPORTED, UNSUPPORTED, '
            'or CONTRADICTED, citing frame positions that decide it. Judge only what is visible. '
            'Return JSON {"claims":[{"claim":"...","verdict":"SUPPORTED",'
            '"frames":"1-2"}]}. Caption: '+caption+'\nFrame positions: '+json.dumps(positions))


def skeptic_failures(verdict):
    claims=verdict.get('claims') if isinstance(verdict,dict) else None
    if not isinstance(claims,list):raise ValueError('bad JSON skeptic claims')
    failures=[]
    for claim in claims:
        if not isinstance(claim,dict) or claim.get('verdict') not in ('SUPPORTED','UNSUPPORTED','CONTRADICTED'):
            raise ValueError('bad JSON skeptic verdict')
        if claim['verdict']!='SUPPORTED':failures.append({'claim':str(claim.get('claim','')),'verdict':claim['verdict'],'frames':str(claim.get('frames',''))})
    return failures


def skeptic_stage(result,ask,positions):
    first=skeptic_failures(ask(positions,{},skeptic_caption=result['caption'],skeptic_request=True))
    if not first:
        result['vidgin_skeptic_flags']=[]
        return result,0
    banned=[x['claim'] for x in first]
    rewrite=ask(positions,{},banned_claims=banned,rewrite_request=True)
    revised=caption_from_observation(rewrite,result['_facts'])
    second=skeptic_failures(ask(positions,{},skeptic_caption=revised['caption'],skeptic_request=True))
    revised['vidgin_skeptic_flags']=second
    revised['caption_needs_review']=revised['caption_needs_review'] or bool(second)
    return revised,1


def ab_pairs(root,reference_dir):
    return benchmark_metrics(root,reference_dir)


def make_ab_pack(root,reference_dir,n,seed=42):
    pairs=ab_pairs(root,reference_dir)
    chosen=random.Random(seed).sample(pairs,min(n,len(pairs)))
    out=Path(root) if Path(root).is_dir() else Path(root).parent
    pack=out/'vidgin_ab_pack.md';key=out/'vidgin_ab_key.json'
    lines=['# Blind video caption ratings','','Rate each caption 1-5 for visible accuracy and useful detail.','']
    mapping={}
    for i,(k,a,b) in enumerate(chosen,1):
        flip=random.Random(seed+i).choice([True,False])
        A,B=(a,b) if flip else (b,a)
        mapping[str(i)]={'clip':f'{k[0]}/{k[1]}','tool':'A' if flip else 'B'}
        lines.extend([f'## Pair {i}: {k[0]}/{k[1]}','',f'Caption A: {A.get("caption","")}','',
                      'Rating A (1-5): ____','Grounding errors A: ____','',
                      f'Caption B: {B.get("caption","")}','','Rating B (1-5): ____','Grounding errors B: ____',''])
    pack.write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(key,{'seed':seed,'pairs':mapping})
    return pack,key


def score_ab(key_path,rated_path):
    key=json.loads(Path(key_path).read_text())['pairs'];txt=Path(rated_path).read_text()
    wins=ties=losses=0;found=0
    for match in re.finditer(r'^## Pair (\d+):.*?(?=^## Pair |\Z)',txt,re.M|re.S):
        i=match.group(1);part=match.group(0)
        a=re.search(r'Rating A \(1-5\):\s*([1-5])',part)
        b=re.search(r'Rating B \(1-5\):\s*([1-5])',part)
        if not a or not b or i not in key:continue
        found+=1;av=int(a.group(1));bv=int(b.group(1));diff=av-bv if key[i]['tool']=='A' else bv-av
        wins+=diff>0;ties+=diff==0;losses+=diff<0
    result={'rated':found,'wins':wins,'ties':ties,'losses':losses}
    print('A/B SCORE',result)
    return result

def png_bytes(rgb):
    # PNG writer in pure stdlib; explicitly RGB, no BGR conversion.
    import struct, zlib
    h,w,_ = rgb.shape
    raw = b''.join(b'\0'+rgb[y].tobytes() for y in range(h))
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b'')


def prompt(facts):
    examples = [
        {"observations":["frames 1-2: a red bus on a road", "frames 3-4: the bus moves to the right"],"subject":"a red bus","action":"moves to the right","setting":"a road","start_state":"the bus is left of center","end_state":"the bus is farther right"},
        {"observations":["frames 1-2: a person sits on a sofa", "frames 3-4: the person remains seated"],"subject":"a person","action":"sits","setting":"a room with a sofa","start_state":"the person is seated","end_state":"the person is still seated"}
    ]
    return ('You annotate actual video frames. Return a JSON object with keys in this order: '
            'observations (list of short visible facts each prefixed with frame positions), '
            'subject, action, setting, start_state, mid_state, end_state, secondary_subjects. All fields MUST be supported '
            'by the observations. Use present tense, no invented objects, lighting or events. '
            'If banned_claims is present, do not state any banned claims. Do not copy the following invented TEACHING EXAMPLES into real clip output: '
            +json.dumps(examples)+'\nMeasured facts (authoritative): '+json.dumps(facts,default=str,allow_nan=False))


class Backend:
    def describe(self, frames, facts): raise NotImplementedError

class StubBackend(Backend):
    def describe(self,frames,facts):
        return {'observations':['frames 1-2: a geometric object on a plain background',
                                'frames 3-4: the geometric object remains on the plain background'],
                'subject':'a geometric object','action':'remains visible',
                'setting':'a plain background','start_state':'the object is visible',
                'end_state':'the object remains visible'}

class OllamaBackend(Backend):
    def __init__(self,model,url):
        self.model=model; self.url=url.rstrip('/'); self.session=requests.Session()
        validate_local(self.url)
    def describe(self,frames,facts):
        images=[f if isinstance(f,str) else base64.b64encode(png_bytes(f)).decode('ascii') for f in frames]
        data={'model':self.model,'stream':False,'format':'json',
              'messages':[{'role':'user','content':skeptic_prompt(facts['skeptic_caption'],facts.get('frame_positions',[])) if facts.get('skeptic_request') else prompt(facts),'images':images}]}
        r=self.session.post(self.url+'/api/chat',json=data,timeout=240); r.raise_for_status()
        return json.loads(r.json()['message']['content'])

class LocalCompatibleBackend(Backend):
    def __init__(self,model,url):
        self.model=model; self.url=url.rstrip('/'); self.session=requests.Session()
        validate_local(self.url)
    def describe(self,frames,facts):
        content=[{'type':'text','text':skeptic_prompt(facts['skeptic_caption'],facts.get('frame_positions',[])) if facts.get('skeptic_request') else prompt(facts)}]
        for f in frames:
            content.append({'type':'image_url','image_url':{'url':'data:image/'+('jpeg' if facts.get('image_format')=='jpeg' else 'png')+';base64,'+(f if isinstance(f,str) else base64.b64encode(png_bytes(f)).decode('ascii'))}})
        r=self.session.post(self.url+'/chat/completions',json={'model':self.model,'messages':[{'role':'user','content':content}], 'temperature':0,'max_tokens':450},timeout=240)
        r.raise_for_status(); s=r.json()['choices'][0]['message']['content']
        return json.loads(s)


def words(s): return re.findall(r"\b[\w'-]+\b",s,flags=re.UNICODE)

def clean(s):
    return re.sub(r'\s+',' ',str(s or 'unknown')).strip().rstrip('.; ')

STOPWORDS=set("a an the and or of in on at to with from is are was were be been by for this that there it its as into over under across sampled frames frame scene set against through remains visible shows show while during about then end start near far left right".split())

def content_tokens(value):
    return {w.lower() for w in words(normalize(str(value))) if len(w)>2 and w.lower() not in STOPWORDS}

def grounding_flags(obs, facts):
    observations=obs.get('observations',[])
    if not isinstance(observations,list) or not observations or any(not isinstance(x,str) for x in observations):
        return ['missing_observations']
    grounded=content_tokens(' '.join(observations))
    # Measurements are facts, not model guesses.
    grounded |= content_tokens(' '.join(str(facts.get(k,'')) for k in ('camera_motion','lighting','subject_distance','subject_placement','subject_side')))
    return sorted((content_tokens(obs.get('subject','')) | content_tokens(obs.get('setting',''))) - grounded)

def caption_from_observation(obs,facts):
    if not isinstance(obs,dict): raise ValueError('Model output must be JSON object')
    for key in ('subject','action','setting','start_state','end_state'):
        if not isinstance(obs.get(key),str) or not obs[key].strip(): raise ValueError('Missing observation '+key)
    subject,action,setting,start,end=(normalize(clean(obs[k]),count=k in ('subject','action','setting')) for k in ('subject','action','setting','start_state','end_state'))
    mid=normalize(clean(obs.get('mid_state',''))) if obs.get('mid_state') else ''
    change=(f'{start}; midway, {mid}; by the final sampled frame, {end}' if mid and content_tokens(mid)!=content_tokens(start) and content_tokens(mid)!=content_tokens(end) else f'{start}; by the final sampled frame, {end}')
    sentences=[f'{subject[0].upper()+subject[1:]} {action}.',f'The scene is set against {setting}.',f'Across the sampled frames, {change}.']
    if facts['near_static']:sentences.append('The sampled frames show very little overall visual change.')
    if facts['scene_cut']:sentences.append('A large abrupt frame difference suggests a possible scene cut.')
    if facts['near_black']:sentences.append('Most sampled pixels are very dark.')
    caption=' '.join(sentences)
    # Round 7: reuse only existing, distinct, model-asserted observations.
    # Preserve in-range drafts byte-for-byte; never truncate or fabricate details.
    if len(words(caption)) < 40:
        for observation in obs.get('observations', []):
            if not isinstance(observation, str):
                continue
            detail = re.sub(r'^\s*frames?\s+\d+(?:\s*[-–,]\s*\d+)*\s*:\s*', '', observation, flags=re.I)
            detail = normalize(clean(detail))
            if not detail or detail.lower() == 'unknown':
                continue
            detail_tokens = content_tokens(detail)
            existing_tokens = content_tokens(caption)
            if not detail_tokens or len(detail_tokens & existing_tokens) / len(detail_tokens) >= 0.8:
                continue
            detail_sentence = detail[0].upper() + detail[1:] + '.'
            # Reject duplicate details, including paraphrases with substantial overlap.
            prior_details = [x for x in sentences[3:] if x]
            if any(detail_tokens == content_tokens(x) or
                   (content_tokens(x) and len(detail_tokens & content_tokens(x)) / len(detail_tokens) >= 0.8)
                   for x in prior_details):
                continue
            if len(words(caption)) + len(words(detail_sentence)) > 70:
                continue
            sentences.append(detail_sentence)
            caption = ' '.join(sentences)
            if len(words(caption)) >= 40:
                break
    if len(words(caption))>90:
        # Trim only the scene-change verbosity; never invent filler.
        keep=90-len(words(' '.join(sentences[:2]+sentences[3:])))
        sentences[2]='Across the sampled frames, '+' '.join(words(change)[:max(0,keep-4)])+'.'
        caption=' '.join(sentences)
    short=f'{subject[0].upper()+subject[1:]} {action} in {setting}.'
    if len(words(short))>25:short=' '.join(words(short)[:25])+'.'
    flags=grounding_flags(obs,facts)
    lint=sorted(set(w.lower() for w in words(caption)) & VAGUE_WORDS)
    return {'caption':caption,'caption_short':short,'subject':subject,'action':action,
            'setting':setting,'scene_change':change,'start_state':start,'mid_state':mid,'end_state':end,
            'secondary_subjects':obs.get('secondary_subjects',[]) if isinstance(obs.get('secondary_subjects',[]),list) else [],
            'vidgin_lint_flags':lint,'vidgin_measurement_confidence':facts.get('vidgin_measurement_confidence',{}),
            'observations':obs['observations'],'vidgin_grounding_flags':flags,
            'camera_motion':facts['camera_motion'] if facts['camera_motion'] in CAMERA else 'unknown',
            'lighting':facts['lighting'] if facts['lighting'] in LIGHT else 'unknown',
            'subject_distance':facts['subject_distance'] if facts['subject_distance'] in DISTANCE else 'unknown',
            'subject_placement':facts['subject_placement'],'scene_cut':facts['scene_cut'],
            'near_static':facts['near_static'],'near_black':facts['near_black'],
            'caption_source':'vlm','caption_needs_review':bool(flags) or bool(lint) or not 40<=len(words(caption))<=90 or facts['near_black'] or facts['scene_cut']}

def candidate_score(result):
    obs=result.get('observations',[])
    tokens=content_tokens(result.get('caption',''))
    coverage=sum(bool(content_tokens(o)&tokens) for o in obs)/max(1,len(obs))
    return round(3*(not result['vidgin_grounding_flags'])+2*(40<=len(words(result['caption']))<=90)+coverage-2*len(result['vidgin_grounding_flags']),4)

def frame_indices(frames, idx, minimum=4, threshold=1.5):
    kept=[]
    for i in idx:
        f=np.asarray(frames[i],dtype=np.int16)
        if not kept or all(np.abs(f-np.asarray(frames[j],dtype=np.int16)).mean()>=threshold for j in kept):kept.append(i)
    if len(kept)<min(minimum,len(idx)):
        for i in idx:
            if i not in kept:kept.append(i)
            if len(kept)>=min(minimum,len(idx)):break
    return sorted(kept)

def upscale(rgb,scale):
    if scale==1:return np.asarray(rgb)
    if cv2 is not None:return cv2.resize(np.asarray(rgb),None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    # NumPy bilinear resampling fallback.
    a=np.asarray(rgb);h,w=a.shape[:2]
    yy=np.arange(h*scale)/scale;xx=np.arange(w*scale)/scale
    y0=np.floor(yy).astype(int);x0=np.floor(xx).astype(int)
    y1=np.minimum(y0+1,h-1);x1=np.minimum(x0+1,w-1)
    fy=(yy-y0)[:,None,None];fx=(xx-x0)[None,:,None]
    result=(a[y0[:,None],x0[None,:]]*(1-fy)*(1-fx)+a[y1[:,None],x0[None,:]]*fy*(1-fx)+a[y0[:,None],x1[None,:]]*(1-fy)*fx+a[y1[:,None],x1[None,:]]*fy*fx)
    return np.clip(result,0,255).astype(np.uint8)

class FrameCache:
    def __init__(self,frames,scale=1,fmt='png'):
        self.frames=frames;self.scale=scale;self.fmt=fmt;self.cache={};self.encodes=0;self.sent=[]
    def encode(self,idx):
        key=(idx,self.scale,self.fmt)
        if key not in self.cache:
            image=upscale(self.frames[idx],self.scale)
            if self.fmt=='jpeg':
                if cv2 is None:raise RuntimeError('JPEG encoding requires optional OpenCV; use --image-format png')
                ok,buf=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
                if not ok:raise ValueError('JPEG encoding failed')
                data=buf.tobytes()
            else:data=png_bytes(image)
            self.cache[key]=base64.b64encode(data).decode('ascii');self.encodes+=1
        return self.cache[key]
    def payload(self,idx):
        chosen=frame_indices(self.frames,idx)
        self.sent.append(len(chosen))
        return [self.encode(i) for i in chosen],chosen


def find_clip(manifest,entry):
    for key in ('clip_path','video_path','path','file','filename'):
        if isinstance(entry.get(key),str):
            p=Path(entry[key]); p=p if p.is_absolute() else manifest.parent/p
            if p.is_file(): return p
    for key in ('clip_id','id'):
        if key in entry:
            val=str(entry[key]); name=val if val.startswith('clip_') else 'clip_'+val.zfill(6)
            for ext in ('.npy','.mp4'):
                p=manifest.parent/(name+ext)
                if p.exists(): return p
    return None


def atomic_lines(path,lines,backup=True):
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.writelines(lines); f.flush(); os.fsync(f.fileno())
        if backup: shutil.copy2(path,str(path)+'.bak')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path,value):
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def backend_identity(backend):
    if isinstance(backend, StubBackend): return 'stub', 'stub'
    if isinstance(backend, OllamaBackend): return 'ollama', backend.model
    if isinstance(backend, LocalCompatibleBackend): return 'compatible', backend.model
    return 'stub', 'stub'  # unknown test backends cannot claim real provenance


def baseline_path(root):
    root=Path(root)
    return (root if root.is_dir() else root.parent)/'.gemini_baseline.json'


def snapshot_baseline(root):
    target=baseline_path(root)
    entries={}
    for m in manifests(root):
        rel=m.relative_to(target.parent).as_posix()
        for i,line in enumerate(m.read_bytes().splitlines(keepends=True)):
            if json.loads(line).get('caption_source')=='gemini':
                entries[f'{rel}::{i}']=hashlib.sha256(line).hexdigest()
    # Exclusive creation; a baseline cannot be silently replaced.
    with target.open('x',encoding='utf-8') as f:
        json.dump(entries,f,indent=2,sort_keys=True)
        f.flush();os.fsync(f.fileno())
    print(f'BASELINE snapshot: {len(entries)} Gemini rows -> {target}')
    return target


def registry_load(path):
    result={}
    path=Path(path)
    if not path.exists():return result
    lines=path.read_bytes().splitlines()
    for i,line in enumerate(lines):
        try:
            row=json.loads(line)
            if row.get('vidgin_backend') in ('stub','ollama','compatible') and row.get('vidgin_hash') and isinstance(row.get('result'),dict):
                result.setdefault(row['vidgin_hash'],row['result'])
        except (ValueError,TypeError):
            if i!=len(lines)-1:raise ValueError(f'Corrupt registry at line {i+1}')
    return result


def registry_append(path,items):
    if not items:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('ab') as f:
        for digest,result in items:
            f.write((json.dumps({'vidgin_hash':digest,'vidgin_backend':result['vidgin_backend'],'result':result},ensure_ascii=False)+'\n').encode())
        f.flush();os.fsync(f.fileno())


class BreakerOpen(Exception):pass

def process_manifest(path,backend,force=False,quiet=False,self_check=False,shard=None,frame_scale=None,image_format='png',skeptic=False,skeptic_backend=None,registry=None,breaker=5,flush_every=10):
    if flush_every<0:raise ValueError('--flush-every must be >= 0')
    if frame_scale is None:frame_scale=1 if isinstance(backend,StubBackend) else 2
    raw=path.read_bytes().splitlines(keepends=True);out=[]
    stats={'processed':0,'skipped':0,'failed':0,'errors':collections.Counter(),
           'load_times':[],'analysis_times':[],'model_times':[],'frames_sent':[],
           'grounding_flags':0,'candidate_margins':[],'image_encodes':0,'normalizations':0,'skeptic_rewrites':0,'reused':0,'breaker_open':False}
    consecutive=0; pending=[]; registry_data=registry_load(registry) if registry else {}
    start=time.monotonic()
    original_backup_done=False
    flushed_count=0
    last_flushed=0
    def write_progress():
        atomic_json(path.parent/'.caption_progress.json',{
            'manifest':str(path.name),'worker_pid':os.getpid(),'processed':stats['processed'],
            'skipped':stats['skipped'],'failed':stats['failed'],'reused':stats['reused'],
            'breaker_open':stats['breaker_open'],'failure_categories':dict(stats['errors'])})
    def flush_manifest(final=False):
        nonlocal original_backup_done,last_flushed
        # A prefix of complete output rows, followed by byte-for-byte original untouched rows.
        current=out+([] if final else raw[len(out):])
        if len(current)!=len(raw):raise AssertionError('incremental flush lost rows')
        if current!=raw and (final or len(out)>last_flushed):
            atomic_lines(path,current,backup=not original_backup_done)
            original_backup_done=True
            last_flushed=len(out)
        if registry is not None and pending:
            registry_append(registry,pending)
            pending.clear()
        write_progress()
    for i,line in enumerate(raw):
        if STOP:out.extend(raw[i:]);break
        try:
            obj=json.loads(line)
            if not isinstance(obj,dict):raise ValueError('Not a JSON object')
            if obj.get('caption_source')=='gemini' and not force:
                out.append(line);stats['skipped']+=1;continue
            if shard is not None:
                key=next((str(obj[k]) for k in ('clip_path','video_path','path','file','filename') if isinstance(obj.get(k),str)),None)
                if key is None and obj.get('clip_id') is not None:key='clip_'+str(obj['clip_id']).zfill(6)+'.npy'
                if key is not None and shard_for(key,shard[1])!=shard[0]:
                    out.append(line);stats['skipped']+=1;continue
            clip=find_clip(path,obj)
            if clip is None:raise FileNotFoundError('Clip not found for manifest entry')
            clip_start=time.monotonic();t=time.monotonic()
            frames,depth=load_clip(clip);digest=canonical_hash(frames,depth)
            if obj.get('caption_source')=='vlm' and obj.get('vidgin_hash')==digest and not force:
                out.append(line);stats['skipped']+=1;continue
            if digest in registry_data and not force:
                reused=registry_data[digest]
                if 'caption_weak' in obj and not isinstance(obj['caption_weak'],str):raise ValueError('caption_weak must be a string')
                obj.update(reused);obj['vidgin_hash']=digest
                out.append((json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n').encode())
                stats['reused']+=1;stats['processed']+=1;consecutive=0
                flushed_count+=1
                if flush_every and flushed_count>=flush_every:
                    flush_manifest();flushed_count=0
                continue
            stats['load_times'].append(time.monotonic()-t);t=time.monotonic()
            selected,facts=analyze(frames,depth)
            facts=confidence_adjust(frames,depth,facts)
            stats['analysis_times'].append(time.monotonic()-t)
            cache=FrameCache(frames,frame_scale,image_format)
            def ask(indices,base,**extra):
                payload,chosen=cache.payload(indices)
                supplied=dict(base,**extra)
                supplied['image_format']=image_format
                supplied['sample_indices']=chosen
                supplied['frame_positions']=[f'frame {j+1} of {len(chosen)}, about {round(100*idx/max(len(frames)-1,1))}% through the clip' for j,idx in enumerate(chosen)]
                return robust_describe(skeptic_backend or backend if supplied.get('skeptic_request') else backend,payload,supplied)
            model_start=time.monotonic()
            obs=ask(facts['sample_indices'],facts)
            result=caption_from_observation(obs,facts)
            if len(words(result['caption']))<40:
                _,extra_facts=analyze(frames,depth,n=16)
                expanded=ask(extra_facts['sample_indices'],extra_facts,expansion_request='Return additional observed facts with frame positions. No padding.')
                if isinstance(expanded,dict):
                    # Only accept expansion facts as observations, and update end state if observed.
                    additional=expanded.get('observations',[])
                    if isinstance(additional,list):
                        obs=dict(obs);obs['observations']=list(obs.get('observations',[]))+[v for v in additional if isinstance(v,str) and v not in obs.get('observations',[])]
                        if expanded.get('end_state') and content_tokens(expanded['end_state']) & content_tokens(' '.join(additional)):
                            obs['end_state']=expanded['end_state']
                        result=caption_from_observation(obs,facts)
            if self_check:
                alternate_idx=sorted(set(range(1,len(frames),max(1,len(frames)//7))))
                check=ask(alternate_idx,facts,self_check=True)
                other=caption_from_observation(check,facts)
                score_a=candidate_score(result);score_b=candidate_score(other)
                if score_b>score_a:result,obs=other,check
                margin=abs(score_a-score_b)
                result['vidgin_candidate_scores']={'candidate_a':score_a,'candidate_b':score_b,
                    'selected':'B' if score_b>score_a else 'A','loser_score':min(score_a,score_b),'margin':margin}
                stats['candidate_margins'].append(margin)
                if margin<0.5:result['caption_needs_review']=True
            if skeptic:
                def skeptic_ask(indices,base,**extra):
                    return ask(indices,dict(facts,**base),**extra)
                result['_facts']=facts
                result,rewrites=skeptic_stage(result,skeptic_ask,facts['sample_indices'])
                stats['skeptic_rewrites']+=rewrites
                result.pop('_facts',None)
            result['caption_needs_review']=result['caption_needs_review'] or not 40<=len(words(result['caption']))<=90
            stats['model_times'].append(time.monotonic()-model_start)
            stats['frames_sent'].extend(cache.sent);stats['image_encodes']+=cache.encodes
            stats['grounding_flags']+=bool(result['vidgin_grounding_flags'])
            if 'caption_weak' in obj and not isinstance(obj['caption_weak'],str):raise ValueError('caption_weak must be a string')
            obj.update(result);obj['vidgin_backend'],obj['vidgin_model']=backend_identity(backend);obj['vidgin_hash']=digest;obj['vidgin_seconds']=round(time.monotonic()-clip_start,4)
            obj['vidgin_frames_sent']=cache.sent;obj['vidgin_image_encodes']=cache.encodes
            out.append((json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n').encode())
            stats['processed']+=1;consecutive=0
            if registry is not None:
                reusable={k:v for k,v in obj.items() if k in result or k in ('vidgin_backend','vidgin_model','vidgin_hash','vidgin_seconds','vidgin_frames_sent','vidgin_image_encodes')}
                registry_data.setdefault(digest,reusable);pending.append((digest,reusable))
            flushed_count+=1
            if flush_every and flushed_count>=flush_every:
                flush_manifest();flushed_count=0
        except Exception as e:
            out.append(line);stats['failed']+=1;category=classify_error(e);stats['errors'][category]+=1
            consecutive=consecutive+1 if category in ('timeout','http','bad_json') else 0
            if breaker and consecutive>=breaker:
                stats['breaker_open']=True
                out.extend(raw[i+1:])
            print(f'FAIL {path}:{i+1}: {type(e).__name__}: {e}',file=sys.stderr)
            if stats['breaker_open']:break
        if not quiet and (i+1)%10==0:
            rate=(time.monotonic()-start)/(i+1)
            print(f'{path.name} {i+1}/{len(raw)} ETA {(len(raw)-i-1)*rate:.0f}s',flush=True)
    flush_manifest(final=True)
    return stats

def manifests(root):
    p=Path(root)
    if p.is_file() and p.name=='manifest.jsonl': return [p]
    return sorted(p.rglob('manifest.jsonl'))


def verify(root):
    problems=[];total=0;checked=0
    bp=baseline_path(root)
    baseline=json.loads(bp.read_text()) if bp.exists() else None
    for m in manifests(root):
        bak=Path(str(m)+'.bak');old=bak.read_bytes().splitlines(keepends=True) if bak.exists() else []
        current=m.read_bytes().splitlines(keepends=True)
        rel=m.relative_to(bp.parent).as_posix()
        for i,line in enumerate(current):
            total+=1
            try:
                e=json.loads(line)
                if 'caption_weak' in e and not isinstance(e['caption_weak'],str):problems.append(f'{m}:{i+1}: caption_weak not string')
                if e.get('caption_source')=='vlm':
                    for k in ('caption','caption_short',*FIELDS,'caption_needs_review','vidgin_hash'):
                        if k not in e:problems.append(f'{m}:{i+1}: missing {k}')
                    if not isinstance(e.get('caption_needs_review'),bool):problems.append(f'{m}:{i+1}: review flag not boolean')
                    n=len(words(e.get('caption','')))
                    if not 40<=n<=90 and not e.get('caption_needs_review'):problems.append(f'{m}:{i+1}: unflagged caption length {n} outside 40..90')
                    if len(words(e.get('caption_short','')))>25:problems.append(f'{m}:{i+1}: short >25 words')
                    if not all(str(e.get(k,'')).strip() for k in FIELDS):problems.append(f'{m}:{i+1}: missing structured content')
                if baseline is not None:
                    key=f'{rel}::{i}'
                    if key in baseline:
                        checked+=1
                        if hashlib.sha256(line).hexdigest()!=baseline[key]:problems.append(f'{m}:{i+1}: Gemini baseline hash mismatch')
                    elif e.get('caption_source')=='gemini':problems.append(f'{m}:{i+1}: Gemini row absent from baseline')
                elif e.get('caption_source')=='gemini' and i<len(old) and line!=old[i]:
                    problems.append(f'{m}:{i+1}: Gemini bytes changed from backup')
            except Exception as ex:problems.append(f'{m}:{i+1}: {ex}')
    if baseline is not None and checked!=len(baseline):problems.append(f'baseline missing rows: expected {len(baseline)}, found {checked}')
    print(f'VERIFY entries={total} violations={len(problems)}')
    print(f'baseline: {checked} rows checked' if baseline is not None else 'baseline: absent; .bak comparison fallback')
    for p in problems[:50]:print('  '+p)
    if baseline is None and not any(Path(str(m)+'.bak').exists() for m in manifests(root)):
        print('NOTE: No backups: Gemini byte identity cannot be proven from current files alone')
    return not problems


def token_f1(a,b):
    from collections import Counter
    x=Counter(w.lower() for w in words(a));y=Counter(w.lower() for w in words(b))
    if not x and not y:return 1.
    if not x or not y:return 0.
    common=sum((x&y).values());p=common/sum(x.values());r=common/sum(y.values())
    return 2*p*r/(p+r) if p+r else 0.

def benchmark(root,reference_dir=None,review_pack=0):
    generated={};reference={}
    def key(m,e,base):
        filename=next((str(e[k]) for k in ('clip_path','video_path','path','file','filename') if isinstance(e.get(k),str)), '')
        return (m.parent.relative_to(Path(base)).as_posix() if Path(base).is_dir() else '.',Path(filename).name)
    for m in manifests(root):
        for line in m.read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e.get('caption_source')!='vlm':continue
            k=key(m,e,root)
            if k[1]:generated[k]=e
            if isinstance(e.get('gemini_reference'),dict):reference[k]=e['gemini_reference']
    if reference_dir:
        reference={}
        for m in manifests(reference_dir):
            for line in m.read_text(encoding='utf-8').splitlines():
                e=json.loads(line);k=key(m,e,reference_dir)
                if k[1]:reference[k]=e
    pairs=[(k,v,reference[k]) for k,v in sorted(generated.items()) if k in reference]
    if not pairs:print('BENCHMARK FAIL: no matching clip filenames');return False
    confusion=collections.Counter()
    for field in FIELDS:
        applicable=[(a,b) for _,a,b in pairs if str(b.get(field,'') or '').strip()]
        if not applicable:
            print(f'{field}: exact=— token_F1=— synonym_F1=— (0 applicable pairs)')
            continue
        exact=100*sum(clean(a.get(field,'')).lower()==clean(b.get(field,'')).lower() for a,b in applicable)/len(applicable)
        f1=statistics.mean(token_f1(str(a.get(field,'')),str(b.get(field,''))) for a,b in applicable)
        synonym=statistics.mean(token_f1(normalize(str(a.get(field,''))),normalize(str(b.get(field,'')))) for a,b in applicable)
        print(f'{field}: exact={exact:.1f}% token_F1={f1:.3f} synonym_F1={synonym:.3f} ({len(applicable)} applicable pairs)')
        if field=='camera_motion':
            for a,b in applicable:confusion[(str(b.get(field,'unknown')),str(a.get(field,'unknown')))]+=1
    print('camera_motion confusion (reference -> generated):',dict(sorted((f'{a} -> {b}',v) for (a,b),v in confusion.items())))
    capf1=statistics.mean(token_f1(str(a.get('caption','')),str(b.get('caption',''))) for _,a,b in pairs)
    synf1=statistics.mean(token_f1(normalize(str(a.get('caption',''))),normalize(str(b.get('caption','')))) for _,a,b in pairs)
    lengths=[len(words(a.get('caption',''))) for _,a,_ in pairs]
    compliance=sum(40<=n<=90 for n in lengths)/len(lengths)
    times=[a['vidgin_seconds'] for _,a,_ in pairs if isinstance(a.get('vidgin_seconds'),(float,int))]
    median=statistics.median(times) if times else None
    print(f'Pairs={len(pairs)} caption_token_F1={capf1:.3f} caption_synonym_F1={synf1:.3f} length_compliance={compliance:.1%} median_words={statistics.median(lengths)} median_seconds_per_clip={median if median is not None else "unavailable"}')
    if review_pack:
        chosen=random.Random(42).sample(pairs,min(review_pack,len(pairs)))
        dest=Path(root) if Path(root).is_dir() else Path(root).parent
        output=dest/'vidgin_review_pack.md'
        lines=['# Vidgin human review pack','',f'Random seed: 42; sampled {len(chosen)} of {len(pairs)} matched clips.','']
        for k,a,b in chosen:
            lines += [f'## {k[0]}/{k[1]}','','**Gemini:** '+str(b.get('caption','')),'','**Vidgin:** '+str(a.get('caption','')),'','| Field | Gemini | Vidgin |','|---|---|---|']
            for field in FIELDS:
                escape=lambda x:str(x).replace('|','\\|').replace('\n',' ')
                lines.append(f'| {field} | {escape(b.get(field,""))} | {escape(a.get(field,""))} |')
            lines+=['','Human rating (1-5): ____','Grounding errors: ____','']
        output.write_text('\n'.join(lines),encoding='utf-8')
        print('REVIEW PACK:',output)
    ok=len(pairs)>=30 and compliance>=.95 and synf1>=.50 and median is not None and median<=10
    print('BENCHMARK '+('PASS' if ok else 'FAIL')+' (>=30 pairs, >=95% length, >=0.50 synonym caption F1, <=10s median; human review required separately)')
    return ok

def benchmark_metrics(root,reference_dir):
    generated={};reference={}
    for base,dest in ((root,generated),(reference_dir,reference)):
        for m in manifests(base):
            for line in m.read_text(encoding='utf-8').splitlines():
                e=json.loads(line)
                if dest is generated and e.get('caption_source')!='vlm':continue
                filename=next((str(e[k]) for k in ('clip_path','video_path','path','file','filename') if isinstance(e.get(k),str)),None)
                if filename:
                    key=(m.parent.relative_to(Path(base)).as_posix(),Path(filename).name)
                    dest[key]=e
    return [(k,a,reference[k]) for k,a in generated.items() if k in reference]

def eval_suite(root,reference_dir,ab_result=None):
    verified=verify(root)
    pairs=benchmark_metrics(root,reference_dir)
    benchmark(root,reference_dir,review_pack=10) if pairs else print('BENCHMARK FAIL: no pairs')
    lengths=[len(words(a.get('caption',''))) for _,a,_ in pairs]
    compliant=sum(40<=n<=90 for n in lengths)/len(lengths) if lengths else 0.
    f1=statistics.mean(token_f1(normalize(a.get('caption','')),normalize(b.get('caption',''))) for _,a,b in pairs) if pairs else 0.
    timings=[a.get('vidgin_seconds') for _,a,_ in pairs]
    timing=statistics.median(timings) if timings and all(isinstance(x,(int,float)) for x in timings) else None
    print('STRATIFIED BY REFERENCE CHUNK:')
    for chunk in sorted(set(k[0] for k,_,_ in pairs)):
        subset=[(a,b) for k,a,b in pairs if k[0]==chunk]
        cf1=statistics.mean(token_f1(normalize(a.get('caption','')),normalize(b.get('caption',''))) for a,b in subset)
        lc=sum(40<=len(words(a.get('caption','')))<=90 for a,b in subset)/len(subset)
        print(f'  {chunk}: n={len(subset)} synonym_F1={cf1:.3f} length_compliance={lc:.1%}')
    # Stub provenance is never accepted, regardless of numerical scores.
    real=bool(pairs) and all(a.get('vidgin_backend') in ('ollama','compatible') and bool(a.get('vidgin_model')) for _,a,_ in pairs)
    gates=[('contract',verified,True),('pair-count',len(pairs),30),('length-compliance',compliant,.95),
           ('synonym-caption-F1',f1,.50),('median-total-seconds',timing,10),('real-backend',real,True)]
    passed=[(v is not None and v<=limit) if name=='median-total-seconds' else v>=limit for name,v,limit in gates]
    ready=all(passed)
    print('READINESS: '+('READY' if ready else 'NOT READY'))
    for (name,value,limit),ok in zip(gates,passed):print(f'  {name}: measured={value} threshold={limit} '+('PASS' if ok else 'FAIL'))
    print('A/B:',Path(ab_result).read_text().strip() if ab_result else 'not yet rated')
    print('HUMAN REVIEW REQUIRED: review-pack judgments are not machine-verifiable.')
    return ready

def suggest_normalizations(root):
    counts=Counter()
    for m in manifests(root):
        for line in m.read_text(encoding='utf-8').splitlines():
            e=json.loads(line)
            if e.get('caption_source')=='vlm':counts.update(content_tokens(e.get('caption','')))
    top={w for w,_ in counts.most_common(30)}
    print('WORD | COUNT | PROPOSED CANONICAL (human fills in)')
    for w,n in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0])):
        if n>=5 and w not in top and w not in NORMALIZATION and w not in NORMALIZATION.values():
            print(f'{w} | {n} | ')

def self_test():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); a=np.zeros((33,256,256,3),dtype=np.uint8)
        for i in range(33): a[i,40:90,20+i:70+i]=[240,60,30]
        np.save(p/'clip_000001.npy',a)
        np.save(p/'clip_000001_depth.npy',np.ones((33,256,256),dtype=np.float32))
        np.save(p/'clip_000002.npy',a)
        original_gemini=b'{"clip_path":"clip_000002.npy", "caption_source":"gemini", "caption_weak":"original gemini"}\n'
        m=p/'manifest.jsonl'
        m.write_bytes(b'{"clip_path":"clip_000001.npy","caption_weak":"weak STRING","other":{"keep":1}}\n'+original_gemini)
        first=process_manifest(m,StubBackend(),quiet=True)
        assert first['processed']==1 and first['skipped']==1 and first['failed']==0
        rows=m.read_bytes().splitlines(keepends=True)
        assert rows[1]==original_gemini
        obj=json.loads(rows[0]); assert obj['caption_weak']=='weak STRING' and obj['other']=={'keep':1}
        assert obj['caption_source']=='vlm' and isinstance(obj['caption_needs_review'],bool)
        assert obj['scene_cut'] is False and len(obj['caption_short'].split())<=25
        assert Path(str(m)+'.bak').exists()
        second=process_manifest(m,StubBackend(),quiet=True)
        assert second['processed']==0 and second['skipped']==2
        assert m.read_bytes().splitlines(keepends=True)==rows
        assert verify(p)
        # Tamper with a Gemini row and confirm verifier catches it.
        m.write_bytes(rows[0]+rows[1].replace(b'original gemini',b'changed gemini'))
        assert not verify(p)
        m.write_bytes(b''.join(rows))
        assert verify(p)
        # Extended measured pan, static, inverse depth, lighting, expansion, self-check, join.
        rng=np.random.default_rng(7)
        base=rng.integers(60,190,(128,128,3),dtype=np.uint8)
        pan=np.stack([np.roll(base,i,axis=1) for i in range(33)])
        _,pf=analyze(pan,None)
        assert pf['camera_motion']=='pan_left',pf
        still=np.repeat(base[None],33,axis=0)
        _,sf=analyze(still,None)
        assert sf['camera_motion']=='static',sf
        d=np.ones((33,128,128),dtype=np.float32)
        d[:,32:96,32:96]=10
        _,df=analyze(still,d)
        assert df['subject_distance']=='near' and df['subject_placement']=='foreground',df
        # Round 2: center crop/resize zoom, off-center moving inverse-depth subject.
        zoom_frames=[]
        for t in range(33):
            factor=1+t*.008
            coord=np.clip(np.rint((np.arange(128)-64)/factor+64).astype(int),0,127)
            zoom_frames.append(base[np.ix_(coord,coord)])
        assert analyze(np.stack(zoom_frames),None)[1]['camera_motion']=='zoom_in'
        moving=np.full((33,128,128,3),90,dtype=np.uint8)
        moving_depth=np.ones((33,128,128),dtype=np.float32)
        for t in range(33):
            x=10+t//3
            moving[t,40:68,x:x+20]=220
            moving_depth[t,40:68,x:x+20]=10
        mf=analyze(moving,moving_depth)[1]
        assert mf['subject_distance']=='near' and mf['subject_placement']=='foreground' and mf['subject_side']=='left',mf
        assert normalize('an automobile and folks on a couch')=='an car and people on a sofa'
        names=[f'clip_{i:06d}.npy' for i in range(20)]
        one={x for x in names if shard_for(x,2)==1};two={x for x in names if shard_for(x,2)==2}
        assert not(one&two) and one|two==set(names)
        assert classify_error(requests.exceptions.Timeout())=='timeout'
        assert classify_error(json.JSONDecodeError('x','x',0))=='bad_json'
        print('ROUND2 PASS: camera-centric pan, zoom_in, off-center moving near/foreground left, synonym normalization, disjoint/exhaustive shards, failure categories')
        assert analyze(np.zeros_like(still),None)[1]['lighting']=='low_light'
        assert analyze(np.full_like(still,245),None)[1]['lighting']=='bright'
        class ExpandingStub(StubBackend):
            def __init__(self):self.calls=0
            def describe(self,frames,facts):
                self.calls+=1
                out=super().describe(frames,facts)
                if self.calls>1:
                    out['observations'].append('frames 3-4: a second visible shape occupies the upper part of the plain background')
                    out['end_state']='the geometric object and a second shape remain visible'
                return out
        ex=ExpandingStub()
        m2=p/'extra'/'manifest.jsonl';m2.parent.mkdir()
        np.save(m2.parent/'clip_000003.npy',still)
        m2.write_text('{"clip_path":"clip_000003.npy","caption_weak":"metadata"}\n')
        st=process_manifest(m2,ex,quiet=True,self_check=True)
        assert st['processed']==1 and ex.calls>=3,st
        assert json.loads(m2.read_text())['caption_source']=='vlm'
        # Paired benchmark joins separate manifest trees by filename.
        r=p/'references';r.mkdir()
        (r/'manifest.jsonl').write_text(json.dumps({'clip_path':'clip_000001.npy','caption':'a geometric object','subject':'a geometric object'})+'\n')
        assert token_f1('a geometric object','a geometric object')==1.
        assert not benchmark(p,r,review_pack=1) # one pair must fail minimum-30 gate
        assert (p/'vidgin_review_pack.md').exists()
        print('EXTENDED PASS: pan_left, static, inverse-depth foreground, low_light, bright, expansion, self-check, separate-manifest benchmark join')
        print('STUB TIMING seconds/clip:',json.loads(m2.read_text())['vidgin_seconds'])
        # Round 7: observation weaving, unchanged in-range drafts, no padding.
        terse={'observations':[
            'frames 1-2: a red bus travels along a broad city street beside several parked cars',
            'frames 2-3: pedestrians wait beside a marked crossing near a row of brick buildings',
            'frames 3-4: the red bus approaches an intersection where a traffic signal hangs overhead',
            'frames 4-5: storefront windows and a narrow sidewalk remain visible behind the moving bus'],
            'subject':'a bus','action':'drives','setting':'a street',
            'start_state':'the bus enters the view','end_state':'the bus moves ahead'}
        wf=dict(sf,near_static=False,scene_cut=False,near_black=False)
        woven=caption_from_observation(terse,wf)
        assert 40<=len(words(woven['caption']))<=70, woven['caption']
        assert any('red bus travels along' in sentence.lower() for sentence in woven['caption'].split('. '))
        woven_details=[woven['caption'].split('A red bus travels along',1)[1]] if 'A red bus travels along' in woven['caption'] else []
        woven_details=['A red bus travels along'+x for x in woven_details]
        source_tokens=[content_tokens(re.sub(r'^frames?\s+\d+(?:\s*[-–,]\s*\d+)*\s*:\s*','',o,flags=re.I)) for o in terse['observations']]
        assert all(any(content_tokens(sentence) <= tokens for tokens in source_tokens) for sentence in woven_details if sentence.strip())
        rich=dict(terse,action='travels slowly past a row of parked vehicles while the road remains visible',
                  setting='a broad city street beside a pedestrian crossing and several brick buildings',
                  start_state='the red bus is approaching the intersection and pedestrians stand beside the crossing',
                  end_state='the red bus continues toward the signal and parked vehicles remain along the road')
        # In-range comparison uses the exact pre-Round-7 composition formula.
        subj,act,setting,start,end=(normalize(clean(rich[k]),count=k in ('subject','action','setting')) for k in ('subject','action','setting','start_state','end_state'))
        old=f'{subj[0].upper()+subj[1:]} {act}. The scene is set against {setting}. Across the sampled frames, {start}; by the final sampled frame, {end}.'
        assert 40<=len(words(old))<=70,(len(words(old)),old)
        assert caption_from_observation(rich,wf)['caption']==old
        exhausted=dict(terse,observations=['frames 1-2: a bus on a street'])
        short=caption_from_observation(exhausted,wf)
        assert len(words(short['caption']))<40 and short['caption_needs_review']
        print('weaving PASS: 40-70 observed words, traceable details, in-range unchanged, exhausted stays short')
        print('ROUND7 STUB median_words=',statistics.median([len(words(woven['caption'])),len(words(caption_from_observation(rich,wf)['caption']))]))
        # Round 3 named tests.
        grounded=StubBackend().describe([],{});bad=dict(grounded,subject='a giraffe')
        assert not grounding_flags(grounded,sf) and 'giraffe' in grounding_flags(bad,sf)
        print('grounding PASS: ungrounded subject flagged, consistent observations accepted')
        class SelectingStub(StubBackend):
            def describe(self,frames,facts):
                o=super().describe(frames,facts)
                if facts.get('self_check'):o['subject']='a giraffe'
                return o
        sm=p/'select'/'manifest.jsonl';sm.parent.mkdir()
        np.save(sm.parent/'clip_000004.npy',still)
        sm.write_text('{"clip_path":"clip_000004.npy","caption_weak":"metadata"}\n')
        selection=process_manifest(sm,SelectingStub(),quiet=True,self_check=True)
        so=json.loads(sm.read_text())
        assert selection['processed']==1 and so['vidgin_candidate_scores']['selected']=='A',so
        print('selection PASS: grounded candidate beats ungrounded alternate')
        cache=FrameCache(still,1)
        payload,idx=cache.payload(list(range(0,33,4)))
        assert len(idx)==4 and cache.encodes==4
        cache.payload(list(range(0,33,4)))
        assert cache.encodes==4 and len(cache.sent)==2
        print('frames PASS: static dedup minimum 4; encoded cache reused across calls')
        import contextlib
        buffer=io.StringIO()
        with contextlib.redirect_stdout(buffer):ready=eval_suite(p,r)
        assert not ready and 'READINESS: NOT READY' in buffer.getvalue() and 'pair-count:' in buffer.getvalue() and 'FAIL' in buffer.getvalue()
        print('evalsuite PASS: NOT READY; pair-count gate fails')
        progress=json.loads((sm.parent/'.caption_progress.json').read_text())
        assert progress['processed']==selection['processed'] and progress['skipped']==selection['skipped'] and progress['failed']==selection['failed']
        print('progress PASS: atomic progress counts match run SUMMARY')
        # Round 4: adversarial skeptic, confidence, composition, blind ratings and jobs.
        class SkepticStub(StubBackend):
            def __init__(self,unsupported):self.unsupported=unsupported;self.rewrites=0;self.checks=0
            def describe(self,frames,facts):
                if facts.get('skeptic_request'):
                    self.checks+=1
                    return {'claims':[{'claim':'a fictitious object','verdict':'UNSUPPORTED' if self.unsupported else 'SUPPORTED','frames':'1-4'}]}
                if facts.get('rewrite_request'):self.rewrites+=1
                return super().describe(frames,facts)
        for unsupported in (True,False):
            sk=SkepticStub(unsupported)
            folder=p/('skeptic_bad' if unsupported else 'skeptic_good');folder.mkdir()
            np.save(folder/'clip_000009.npy',still)
            (folder/'manifest.jsonl').write_text('{"clip_path":"clip_000009.npy","caption_weak":"metadata"}\n')
            ss=process_manifest(folder/'manifest.jsonl',sk,quiet=True,skeptic=True)
            row=json.loads((folder/'manifest.jsonl').read_text())
            assert ss['processed']==1 and sk.rewrites==int(unsupported) and sk.checks==(2 if unsupported else 1)
            assert bool(row['vidgin_skeptic_flags'])==unsupported
            if unsupported:assert row['caption_needs_review']
        print('skeptic PASS: unsupported claim triggers exactly one rewrite and persistent flag; supported draft untouched')
        textureless=np.full((33,128,128,3),90,dtype=np.uint8)
        textureless=np.clip(textureless.astype(np.int16)+np.random.default_rng(91).integers(-2,3,textureless.shape),0,255).astype(np.uint8)
        conf=confidence_adjust(textureless,None,analyze(textureless,None)[1])
        assert conf['vidgin_measurement_confidence']['motion']<CONFIDENCE_THRESHOLD['motion'] and conf['camera_motion']=='unknown',conf
        print('confidence PASS: low-texture noise -> low motion confidence and unknown camera motion')
        observation=StubBackend().describe([],{})
        mid=dict(observation,mid_state='the object occupies the middle of the plain background')
        mid_result=caption_from_observation(mid,sf)
        assert 'midway,' in mid_result['scene_change']
        vague=dict(observation,setting='a nice plain background')
        vague_result=caption_from_observation(vague,sf)
        assert 'nice' in vague_result['vidgin_lint_flags'] and 'nice' in vague_result['caption']
        print('composition PASS: three-beat change and vague-word lint without rewrite')
        pack,key=make_ab_pack(p,r,1)
        pack_text=pack.read_text()
        assert 'Gemini' not in pack_text and 'Vidgin' not in pack_text and len(json.loads(key.read_text())['pairs'])==1
        rated=pack_text.replace('Rating A (1-5): ____','Rating A (1-5): 5').replace('Rating B (1-5): ____','Rating B (1-5): 2')
        pack.write_text(rated)
        score=score_ab(key,pack)
        expected_win=json.loads(key.read_text())['pairs']['1']['tool']=='A'
        assert score['rated']==1 and score['wins']==int(expected_win) and score['losses']==int(not expected_win)
        print('abpack PASS: blind labels, complete separate key, expected rated win/tie/loss')
        def build_tree(root):
            for j in range(4):
                folder=root/f'chunk_{j}';folder.mkdir(parents=True)
                np.save(folder/f'clip_{j:06d}.npy',still)
                (folder/'manifest.jsonl').write_text(json.dumps({'clip_path':f'clip_{j:06d}.npy','caption_weak':'keep'})+'\n')
        seq=p/'seq';par=p/'par';build_tree(seq);build_tree(par)
        opts=dict(force=False,self_check=False,shard=None,frame_scale=1,image_format='png',skeptic=False,skeptic_model=None)
        sequential=[manifest_worker((m,'stub','unused','http://127.0.0.1:11434',opts)) for m in manifests(seq)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            parallel=list(pool.map(manifest_worker,[(m,'stub','unused','http://127.0.0.1:11434',opts) for m in manifests(par)]))
        for field in ('processed','skipped','failed'):
            assert sum(x[field] for x in sequential)==sum(x[field] for x in parallel)
        for m in manifests(par):
            progress=json.loads((m.parent/'.caption_progress.json').read_text())
            assert progress['processed']==1 and isinstance(progress['worker_pid'],int)
        assert len({str(m) for m in manifests(par)})==4
        print('parallel PASS: four manifests jobs=2 match sequential counts; one progress writer per manifest')
        print('SELF-TEST PASS: discovery, RGB/depth analysis, stub, atomic backup, preservation, Gemini byte identity, hash resume, verifier detection')
        return True


def round5_tests():
    import contextlib
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);a=np.zeros((33,32,32,3),dtype=np.uint8);a[:,8:24,8:24]=[180,60,20]
        np.save(root/'clip_000001.npy',a)
        m=root/'manifest.jsonl'
        m.write_text(json.dumps({'file':'clip_000001.npy','caption_source':'gemini','caption_weak':'original'})+'\n')
        snapshot_baseline(root)
        try:snapshot_baseline(root);assert False
        except FileExistsError:pass
        original=m.read_bytes()
        # Rotate .bak twice while protected row is present.
        for _ in range(2):
            m.write_bytes(original+b'\n');m.write_bytes(original)
            shutil.copy2(m,str(m)+'.bak')
        m.write_bytes(original.replace(b'original',b'tampered'))
        with contextlib.redirect_stdout(io.StringIO()) as cap:assert not verify(root)
        assert 'baseline hash mismatch' in cap.getvalue()
        m.write_bytes(original)
        print('baseline PASS: immutable snapshot detects tamper after backup rotation')

        # Missing legacy provenance cannot pass real-backend gate.
        gen=root/'gen';ref=root/'ref';gen.mkdir();ref.mkdir()
        g={'file':'clip_000001.npy','caption_source':'vlm','caption':'car on a road','vidgin_seconds':1}
        r={'file':'clip_000001.npy','caption_source':'gemini','caption':'car on a road'}
        (gen/'manifest.jsonl').write_text(json.dumps(g)+'\n')
        (ref/'manifest.jsonl').write_text(json.dumps(r)+'\n')
        with contextlib.redirect_stdout(io.StringIO()) as cap:assert not eval_suite(gen,ref)
        assert 'real-backend: measured=False' in cap.getvalue()
        print('provenance PASS: missing provenance fails real-backend gate')
        with contextlib.redirect_stdout(io.StringIO()) as cap:benchmark(gen,ref)
        out=cap.getvalue()
        assert all(f'{field}: exact=—' in out and '(0 applicable pairs)' in out for field in FIELDS)
        assert 'caption_token_F1=' in out
        print('fieldna PASS: five empty reference fields N/A; caption F1 computed')

        # Cross-manifest registry reuse with different clip filenames.
        reg=root/'registry.jsonl';aa=root/'A';bb=root/'B';aa.mkdir();bb.mkdir()
        np.save(aa/'clip_000010.npy',a);np.save(bb/'clip_000099.npy',a)
        ma=aa/'manifest.jsonl';mb=bb/'manifest.jsonl'
        ma.write_text('{"file":"clip_000010.npy"}\n');mb.write_text('{"file":"clip_000099.npy"}\n')
        class Counting(StubBackend):
            def __init__(self):self.calls=0
            def describe(self,frames,facts):self.calls+=1;return super().describe(frames,facts)
        b=Counting();process_manifest(ma,b,quiet=True,registry=reg);before=b.calls
        st=process_manifest(mb,b,quiet=True,registry=reg)
        assert st['reused']==1 and b.calls==before
        assert json.loads(ma.read_text())['caption']==json.loads(mb.read_text())['caption']
        with reg.open('ab') as f:f.write(b'{truncated')
        assert registry_load(reg)
        print('registry PASS: identical bytes reuse result; no additional backend calls; truncated tail tolerated')

        # Breaker: exactly threshold backend failures, remaining raw rows untouched.
        fail=root/'fail';fail.mkdir()
        for j in range(7):np.save(fail/f'clip_{j:06d}.npy',a)
        fm=fail/'manifest.jsonl';raw=b''.join((json.dumps({'file':f'clip_{j:06d}.npy'})+'\n').encode() for j in range(7));fm.write_bytes(raw)
        class Timeout(StubBackend):
            def __init__(self):self.calls=0
            def describe(self,frames,facts):self.calls+=1;raise requests.exceptions.Timeout('simulated')
        b=Timeout()
        # robust_describe retries internally; breaker counts clip-level failures.
        st=process_manifest(fm,b,quiet=True,breaker=2)
        assert st['failed']==2 and st['breaker_open'] and st['processed']==0
        assert fm.read_bytes()==raw
        print('breaker PASS: threshold=2 failed clips; remaining bytes untouched')

        # Engine check is isolated in temporary directory; existing manifest unchanged.
        before=ma.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as cap:assert engine_check('stub','stub','http://127.0.0.1:11434')==0
        assert 'ENGINE CHECK: OK' in cap.getvalue() and ma.read_bytes()==before
        print('preflight PASS: stub engine check OK; zero existing manifest writes')

        # Exercise both actual HTTP backend implementations using encoded FrameCache strings.
        # No external network: a loopback ephemeral-port server mimics both native APIs.
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        captured=[]
        class FakeVisionServer(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                length=int(self.headers.get('Content-Length','0'))
                request=json.loads(self.rfile.read(length))
                captured.append((self.path,request))
                response=StubBackend().describe([], {})
                if self.path=='/api/chat':
                    assert request['messages'][0]['images']
                    assert all(isinstance(x,str) for x in request['messages'][0]['images'])
                    body={'message':{'content':json.dumps(response)}}
                elif self.path=='/chat/completions':
                    images=[c['image_url']['url'] for c in request['messages'][0]['content'] if c['type']=='image_url']
                    assert images and all(x.startswith('data:image/png;base64,') for x in images)
                    assert all(base64.b64decode(x.split(',',1)[1]).startswith(b'\x89PNG\r\n\x1a\n') for x in images)
                    body={'choices':[{'message':{'content':json.dumps(response)}}]}
                else:
                    self.send_error(404);return
                data=json.dumps(body).encode('utf-8')
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),FakeVisionServer)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for kind,cls in [('ollama',OllamaBackend),('compatible',LocalCompatibleBackend)]:
                folder=root/('http_'+kind);folder.mkdir()
                np.save(folder/'clip_000001.npy',a)
                manifest=folder/'manifest.jsonl';manifest.write_text('{"file":"clip_000001.npy"}\n')
                backend=cls('test-model',f'http://127.0.0.1:{server.server_port}')
                result=process_manifest(manifest,backend,quiet=True)
                assert result['processed']==1 and result['failed']==0,(kind,result)
                row=json.loads(manifest.read_text())
                assert row['caption_source']=='vlm' and row['vidgin_backend']==kind
                assert any(path==('/api/chat' if kind=='ollama' else '/chat/completions') for path,_ in captured)
            assert any(path=='/chat/completions' and any(c.get('image_url',{}).get('url','').startswith('data:image/png;base64,') for c in request['messages'][0]['content']) for path,request in captured)
            print('httpbackends PASS: Ollama and compatible loopback HTTP paths write vlm rows; compatible sends PNG data-URI images')
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)



def round6_tests():
    # Exercise SIGKILL recovery and HTTP follow-up requests.
    import subprocess, threading, contextlib
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        # The subprocess is killed only after an incremental checkpoint has been observed.
        folder=root/'crash';folder.mkdir()
        arr=np.zeros((33,32,32,3),dtype=np.uint8)
        arr[:,7:25,7:25]=[185,60,25]
        for j in range(24):
            frame=arr.copy();frame[:,8:14,8:14,0]=(j*15)%256
            np.save(folder/f'clip_{j:06d}.npy',frame)
        manifest=folder/'manifest.jsonl'
        original=[(json.dumps({'file':f'clip_{j:06d}.npy','caption_weak':'weak'})+'\n').encode() for j in range(24)]
        manifest.write_bytes(b''.join(original))
        worker_code='''
import sys,time,importlib
sys.path.insert(0,sys.argv[1])
v=importlib.import_module(sys.argv[3])
from pathlib import Path
class Slow(v.StubBackend):
 def describe(self,frames,facts):
  time.sleep(.3)
  return super().describe(frames,facts)
v.process_manifest(Path(sys.argv[2]),Slow(),quiet=True,flush_every=2)
'''
        cmd=[sys.executable,'-c',worker_code,str(Path(__file__).resolve().parent),str(manifest),Path(__file__).stem]
        proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+35
            while time.monotonic()<deadline:
                if proc.poll() is not None:raise AssertionError('worker finished before kill checkpoint')
                progress=folder/'.caption_progress.json'
                if progress.exists():
                    try:
                        if json.loads(progress.read_text()).get('processed',0)>=2:break
                    except (ValueError,OSError):pass
                time.sleep(.04)
            else:raise AssertionError('incremental checkpoint not reached')
            proc.kill();proc.wait(timeout=5)
            persisted=manifest.read_bytes().splitlines(keepends=True)
            assert len(persisted)==24 and 2<=sum(json.loads(x).get('caption_source')=='vlm' for x in persisted)<24
            first_count=sum(json.loads(x).get('caption_source')=='vlm' for x in persisted)
            assert all(x==original[i] for i,x in enumerate(persisted) if json.loads(x).get('caption_source')!='vlm')
            assert Path(str(manifest)+'.bak').read_bytes()==b''.join(original)
            resumed=process_manifest(manifest,StubBackend(),quiet=True,flush_every=2)
            assert resumed['processed']+first_count==24 and resumed['failed']==0
            assert resumed['skipped']==first_count
            assert len(manifest.read_bytes().splitlines())==24
            assert all(json.loads(x).get('vidgin_hash') for x in manifest.read_bytes().splitlines())
            assert Path(str(manifest)+'.bak').read_bytes()==b''.join(persisted), 'resume backup is pre-resume state'
            print('flushkill PASS: SIGKILL leaves complete JSONL, original rows preserved; hash resume skips persisted rows')
        finally:
            if proc.poll() is None:proc.kill();proc.wait(timeout=5)
        # A request-aware server: no reliance on request ordinal.
        seen=[]
        class Fake(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/api/chat':
                    msg=data['messages'][0]['content']
                elif self.path=='/chat/completions':
                    msg=data['messages'][0]['content'][0]['text']
                else:self.send_error(404);return
                scenario=self.server.scenario
                repair='\"repair_request\"' in msg
                alternate='\"self_check\"' in msg
                skeptic='INDEPENDENT ADVERSARIAL VISUAL CHECK' in msg
                rewrite='\"rewrite_request\"' in msg
                seen.append((scenario,self.path,repair,alternate,skeptic,rewrite))
                if skeptic:
                    result={'claims':[{'claim':'a forbidden claim','verdict':'UNSUPPORTED','frames':'1-4'}]}
                elif scenario=='repair' and not repair and not alternate and not rewrite:
                    result=None
                else:
                    result=StubBackend().describe([], {})
                    if scenario=='selection' and not alternate:
                        result['subject']='imaginary unicorn'
                    if scenario=='skeptic' and rewrite:
                        result['action']='remains visible'
                payload=json.dumps(result) if result is not None else '{invalid JSON'
                body={'message':{'content':payload}} if self.path=='/api/chat' else {'choices':[{'message':{'content':payload}}]}
                b=json.dumps(body).encode();self.send_response(200)
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)))
                self.end_headers();self.wfile.write(b)
        server=ThreadingHTTPServer(('127.0.0.1',0),Fake)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        try:
            for kind,cls in [('ollama',OllamaBackend),('compatible',LocalCompatibleBackend)]:
                for scenario in ('repair','selection','skeptic'):
                    server.scenario=scenario
                    path=root/(kind+'_'+scenario);path.mkdir()
                    np.save(path/'clip_000001.npy',arr)
                    m=path/'manifest.jsonl';m.write_text('{"file":"clip_000001.npy"}\n')
                    b=cls('test',f'http://127.0.0.1:{server.server_port}')
                    result=process_manifest(m,b,quiet=True,self_check=scenario=='selection',skeptic=scenario=='skeptic')
                    assert result['processed']==1 and result['failed']==0,(kind,scenario,result)
                    row=json.loads(m.read_text())
                    assert row['caption_source']=='vlm'
                    events=[e for e in seen if e[0]==scenario and e[1]==('/api/chat' if kind=='ollama' else '/chat/completions')]
                    if scenario=='repair':assert any(e[2] for e in events),('repair re-ask not sent',events)
                    if scenario=='selection':
                        assert row['vidgin_candidate_scores']['selected']=='B' and row['subject']!='imaginary unicorn'
                        assert any(e[3] for e in events)
                    if scenario=='skeptic':
                        assert result['skeptic_rewrites']==1 and row['vidgin_skeptic_flags'] and row['caption_needs_review']
                        assert sum(e[4] for e in events)==2 and sum(e[5] for e in events)==1
            print('httprecovery PASS: JSON repair, candidate selection, skeptic rewrite via both HTTP backends')
        finally:
            server.shutdown();server.server_close();t.join(timeout=3)

def manifest_worker(arguments):
    path,kind,model,url,opts=arguments
    backend={'stub':lambda:StubBackend(),'ollama':lambda:OllamaBackend(model,url),
             'compatible':lambda:LocalCompatibleBackend(model,url)}[kind]()
    skeptic_backend=None
    if opts.get('skeptic_model') and kind!='stub':
        skeptic_backend={'ollama':lambda:OllamaBackend(opts['skeptic_model'],url),
                         'compatible':lambda:LocalCompatibleBackend(opts['skeptic_model'],url)}[kind]()
    before=NORMALIZATION_COUNT
    result=process_manifest(path,backend,**{k:v for k,v in opts.items() if k!='skeptic_model'},skeptic_backend=skeptic_backend)
    result['normalizations']=NORMALIZATION_COUNT-before
    return result

def engine_check(kind,model,url,scale=None,image_format='png'):
    try:
        if kind=='ollama':
            r=requests.get(url.rstrip('/')+'/api/tags',timeout=10);r.raise_for_status()
            available=[x.get('name','') for x in r.json().get('models',[])]
            if model not in available:print('ENGINE CHECK: model missing',model,'available=',available);return 1
        elif kind=='compatible':
            r=requests.get(url.rstrip('/')+'/models',timeout=10);r.raise_for_status()
            available=[x.get('id','') for x in r.json().get('data',[])]
            if model not in available:print('ENGINE CHECK: model missing',model,'available=',available);return 1
        backend={'stub':lambda:StubBackend(),'ollama':lambda:OllamaBackend(model,url),
                 'compatible':lambda:LocalCompatibleBackend(model,url)}[kind]()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);arr=np.zeros((33,64,64,3),dtype=np.uint8);arr[:,20:40,20:40]=[200,80,30]
            np.save(p/'clip_000001.npy',arr)
            m=p/'manifest.jsonl';m.write_text('{"file":"clip_000001.npy"}\n')
            stats=process_manifest(m,backend,quiet=True,frame_scale=scale,image_format=image_format)
            if stats['failed'] or not stats['processed']:print('ENGINE CHECK: bad output',stats['errors']);return 1
            row=json.loads(m.read_text().splitlines()[0]);print('ENGINE CHECK timing: load+hash=',stats['load_times'],'analysis=',stats['analysis_times'],'model=',stats['model_times'])
            print('ENGINE CHECK words=',len(words(row['caption'])),'grounding=',row['vidgin_grounding_flags'])
            print('ENGINE CHECK: OK');return 0
    except requests.exceptions.RequestException as e:
        print('ENGINE CHECK: unreachable',str(e));return 1
    except Exception as e:
        print('ENGINE CHECK: bad output',str(e));return 1

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',nargs='?',default='.')
    ap.add_argument('--backend',choices=['ollama','compatible','stub'],default='ollama')
    ap.add_argument('--model',default='qwen3-vl:4b-instruct')
    ap.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--snapshot-baseline',metavar='DIR')
    ap.add_argument('--registry',metavar='FILE')
    ap.add_argument('--breaker',type=int,default=5)
    ap.add_argument('--flush-every',type=int,default=10)
    ap.add_argument('--engine-check',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--verify',metavar='DIR')
    ap.add_argument('--benchmark',nargs=2,metavar=('GENERATED_DIR','REFERENCE_DIR'))
    ap.add_argument('--self-check',action='store_true')
    ap.add_argument('--skeptic',action='store_true')
    ap.add_argument('--skeptic-model')
    ap.add_argument('--jobs',type=int,default=1)
    ap.add_argument('--ab-pack',type=int,default=0)
    ap.add_argument('--ab-reference',metavar='REFERENCE_DIR',help='Standalone --ab-pack reference tree; generated tree is positional root')
    ap.add_argument('--ab-score',nargs=2,metavar=('KEY','RATED_PACK'))
    ap.add_argument('--ab-result')
    ap.add_argument('--shard',type=str,help='1-based K/N shard')
    ap.add_argument('--review-pack',type=int,default=0)
    ap.add_argument('--eval-suite',nargs=2,metavar=('GENERATED_DIR','REFERENCE_DIR'))
    ap.add_argument('--suggest-normalizations',metavar='DIR')
    ap.add_argument('--frame-scale',type=int,default=None)
    ap.add_argument('--image-format',choices=['png','jpeg'],default='png')
    args=ap.parse_args()
    if args.self_test:
        ok=self_test()
        if ok:
            round5_tests()
            round6_tests()
        return 0 if ok else 1
    if args.snapshot_baseline:
        try:snapshot_baseline(args.snapshot_baseline);return 0
        except FileExistsError:print('BASELINE ERROR: immutable snapshot already exists',file=sys.stderr);return 1
    if args.ab_score:score_ab(*args.ab_score);return 0
    if args.ab_pack and args.ab_reference:
        pack,key=make_ab_pack(args.root,args.ab_reference,args.ab_pack)
        print('A/B PACK',pack,'KEY',key)
        return 0
    if args.eval_suite:
        if args.ab_pack:make_ab_pack(*args.eval_suite,args.ab_pack)
        return 0 if eval_suite(*args.eval_suite,ab_result=args.ab_result) else 1
    if args.suggest_normalizations:suggest_normalizations(args.suggest_normalizations);return 0
    if args.frame_scale is not None and not 1<=args.frame_scale<=8:ap.error('--frame-scale must be 1..8')
    if args.verify: return 0 if verify(args.verify) else 1
    if args.benchmark:
        if args.ab_pack:make_ab_pack(*args.benchmark,args.ab_pack)
        return 0 if benchmark(*args.benchmark,review_pack=args.review_pack) else 1
    shard=None
    if args.shard:
        try:
            k,n=map(int,args.shard.split('/'));assert 1<=k<=n;shard=(k,n)
        except Exception:ap.error('--shard must be K/N with 1 <= K <= N')
    if args.jobs<1:ap.error('--jobs must be >= 1')
    if args.breaker<0:ap.error('--breaker must be >= 0')
    if args.flush_every<0:ap.error('--flush-every must be >= 0')
    if args.registry and args.jobs>1:ap.error('--registry requires --jobs 1 (append-only single writer)')
    signal.signal(signal.SIGINT,stop_handler)
    paths=manifests(args.root)
    if not paths and not args.engine_check: print('No manifest.jsonl files found',file=sys.stderr);return 1
    url=args.ollama_url if args.backend=='ollama' else os.environ.get('VIDGIN_LOCAL_BASE_URL','http://127.0.0.1:8000/v1')
    if args.backend!='stub':validate_local(url)
    if args.engine_check:return engine_check(args.backend,args.model,url,args.frame_scale,args.image_format)
    opts=dict(force=args.force,self_check=args.self_check,shard=shard,frame_scale=args.frame_scale,
              image_format=args.image_format,skeptic=args.skeptic,skeptic_model=args.skeptic_model,registry=args.registry,breaker=args.breaker,flush_every=args.flush_every)
    tasks=[(m,args.backend,args.model,url,opts) for m in paths]
    results=[]
    if args.jobs==1:
        for task in tasks:
            if STOP:break
            result=manifest_worker(task);results.append(result)
            if result['breaker_open']:break
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures=[pool.submit(manifest_worker,task) for task in tasks]
            try:
                for f in concurrent.futures.as_completed(futures):results.append(f.result())
            except KeyboardInterrupt:
                for f in futures:f.cancel()
                print('Interrupted: waiting for active manifest workers',file=sys.stderr)
    totals={'processed':0,'skipped':0,'failed':0,'reused':0};errors=collections.Counter()
    timings={'load_times':[],'analysis_times':[],'model_times':[],'frames_sent':[],'candidate_margins':[]}
    grounding_total=encodes_total=normalizations=rewrites=0
    for result in results:
        grounding_total+=result['grounding_flags'];encodes_total+=result['image_encodes']
        normalizations+=result['normalizations'];rewrites+=result['skeptic_rewrites']
        errors.update(result['errors'])
        for k in timings:timings[k].extend(result[k])
        for k in totals:totals[k]+=result[k]
    print('SUMMARY',totals,'interrupted='+str(STOP),'normalization_applications='+str(normalizations),'failures_by_category='+str(dict(errors)),'grounding_flag_count='+str(grounding_total),'image_encodes='+str(encodes_total),'skeptic_rewrites='+str(rewrites))
    if any(r['breaker_open'] for r in results):print(f'CIRCUIT BREAKER: backend unreachable after {args.breaker} consecutive failures')
    print('STATS frames_sent_median=',statistics.median(timings['frames_sent']) if timings['frames_sent'] else None,'candidate_win_margin_median=',statistics.median(timings['candidate_margins']) if timings['candidate_margins'] else None)
    print('TIMING median_load_hash_s=',statistics.median(timings['load_times']) if timings['load_times'] else None,'median_analysis_s=',statistics.median(timings['analysis_times']) if timings['analysis_times'] else None,'median_model_s=',statistics.median(timings['model_times']) if timings['model_times'] else None)
    return 1 if totals['failed'] or any(r['breaker_open'] for r in results) else 0

if __name__=='__main__': sys.exit(main())
