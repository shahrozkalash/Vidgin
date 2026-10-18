"""Fixed Validation Prompts Catalog for Video DiT Evaluation (Part 10).

Requirements:
- 50 to 200 curated validation prompts covering 7 distinct axes:
  1. motion: fluid dynamics, human gait, dancing, high-speed physics, particle drift
  2. camera: drone tracking, dolly zoom, 360 orbit, pan left/right, tilt, crane shot
  3. counting: exact item counts (e.g. 3 apples, 5 sunflowers, flock of 8 birds)
  4. text: legible typography, glowing signs, chalkboard, banner lettering
  5. people: facial expressions, intricate hand gestures, complex human movements
  6. animals: quadrupeds, marine creatures, avian flight, insect micro-motion
  7. styles: photorealistic 35mm film, anime/cel-shaded, claymation, cyberpunk, oil painting
- Deterministic fixed seeds for reproducible checkpoint-to-checkpoint benchmarking.
- Structured metadata for evaluation filtering and VBench categorization.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class ValidationPrompt:
    id: str
    prompt: str
    category: str  # 'motion' | 'camera' | 'counting' | 'text' | 'people' | 'animals' | 'styles'
    seed: int
    difficulty: str  # 'standard' | 'challenging' | 'stress_test'
    target_metric: str  # 'motion_smoothness' | 'dynamic_degree' | 'text_alignment' | 'subject_consistency' | 'aesthetic'
    description: str


# Fixed catalog of 70 comprehensive benchmark prompts
VALIDATION_PROMPTS_CATALOG: List[ValidationPrompt] = [
    # 1. Motion & Physics Dynamics (10 prompts)
    ValidationPrompt(
        id="motion_01",
        prompt="A heavy crystal glass dropping into a pool of turquoise water, capturing ultra slow-motion splashing droplets and ripple shockwaves.",
        category="motion",
        seed=1001,
        difficulty="challenging",
        target_metric="dynamic_degree",
        description="Liquid surface tension, splashing fluid dynamics, high frame-to-frame velocity.",
    ),
    ValidationPrompt(
        id="motion_02",
        prompt="A professional ballerina performing a rapid pirouette on a wooden stage, silk dress gracefully billowing with rotational momentum.",
        category="motion",
        seed=1002,
        difficulty="standard",
        target_metric="motion_smoothness",
        description="Rotational kinematic smoothness, fabric motion, temporal coherence.",
    ),
    ValidationPrompt(
        id="motion_03",
        prompt="Dense multicolored smoke exploding outward from a central point against a black backdrop, billowing in turbulent convective currents.",
        category="motion",
        seed=1003,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Non-rigid particle turbulence, volumetric density flow.",
    ),
    ValidationPrompt(
        id="motion_04",
        prompt="A row of red dominoes toppling one after another across a polished marble countertop in cascading sequence.",
        category="motion",
        seed=1004,
        difficulty="challenging",
        target_metric="motion_smoothness",
        description="Causal sequential collision physics, rigid body momentum.",
    ),
    ValidationPrompt(
        id="motion_05",
        prompt="Pouring viscous golden honey in a continuous ribbon onto a fluffy stack of pancakes, honey folding onto itself.",
        category="motion",
        seed=1005,
        difficulty="standard",
        target_metric="motion_smoothness",
        description="High-viscosity fluid coiling, reflective specular highlights.",
    ),
    ValidationPrompt(
        id="motion_06",
        prompt="A martial artist performing an explosive spinning hook kick in the rain, water droplets spraying off the boots.",
        category="motion",
        seed=1006,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Fast limb velocity, multi-emitter particle spray, dynamic motion blur.",
    ),
    ValidationPrompt(
        id="motion_07",
        prompt="Autumn maple leaves drifting lazily in a gentle swirling breeze, floating down towards a cobblestone path.",
        category="motion",
        seed=1007,
        difficulty="standard",
        target_metric="motion_smoothness",
        description="Low-speed aerodynamic flutter, gentle 3D spatial drift.",
    ),
    ValidationPrompt(
        id="motion_08",
        prompt="A sports car drifting around an asphalt hairpin turn, tires smoking and sparks flying from the chassis.",
        category="motion",
        seed=1008,
        difficulty="challenging",
        target_metric="dynamic_degree",
        description="Vehicular kinematics, tire friction smoke, camera-tracking parallax.",
    ),
    ValidationPrompt(
        id="motion_09",
        prompt="A sparkler emitting brilliant golden sparks dancing in the dark on New Year's Eve, illuminating smiling faces.",
        category="motion",
        seed=1009,
        difficulty="standard",
        target_metric="dynamic_degree",
        description="High-contrast particle emission, temporal lighting fluctuations.",
    ),
    ValidationPrompt(
        id="motion_10",
        prompt="A grandfather clock with its pendulum swinging rhythmically back and forth in an antique study.",
        category="motion",
        seed=1010,
        difficulty="standard",
        target_metric="motion_smoothness",
        description="Periodic harmonic motion, rigid object alignment.",
    ),

    # 2. Camera Trajectories & Kinematics (10 prompts)
    ValidationPrompt(
        id="camera_01",
        prompt="Dramatic Hitchcock dolly zoom vertigo effect on a solitary person standing at the edge of a high skyscraper roof.",
        category="camera",
        seed=2001,
        difficulty="stress_test",
        target_metric="background_consistency",
        description="Simultaneous forward dolly and focal zoom in opposite directions.",
    ),
    ValidationPrompt(
        id="camera_02",
        prompt="Continuous 360-degree orbit camera shot smoothly gliding around an ancient Roman marble statue in a sunlit rotunda.",
        category="camera",
        seed=2002,
        difficulty="challenging",
        target_metric="subject_consistency",
        description="Full multi-angle view, specular shifts, 3D geometry consistency.",
    ),
    ValidationPrompt(
        id="camera_03",
        prompt="High-speed FPV drone dive soaring down a sheer vertical alpine cliff edge, skimming over pine treetops into a glacial valley.",
        category="camera",
        seed=2003,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Extreme forward translation velocity, rapid optical flow across scales.",
    ),
    ValidationPrompt(
        id="camera_04",
        prompt="Smooth lateral camera pan from left to right along a bustling neon-lit cyberpunk alleyway in the rain.",
        category="camera",
        seed=2004,
        difficulty="standard",
        target_metric="background_consistency",
        description="Pure horizontal optical flow, background parallax layers.",
    ),
    ValidationPrompt(
        id="camera_05",
        prompt="Slow cinematic crane tilt upwards from wet cobblestones to reveal a towering Gothic cathedral spire against storm clouds.",
        category="camera",
        seed=2005,
        difficulty="standard",
        target_metric="background_consistency",
        description="Vertical translation and pitch change, perspective geometry shift.",
    ),
    ValidationPrompt(
        id="camera_06",
        prompt="Static tripod shot of a bustling Shibuya crosswalk at dusk, pedestrians moving briskly while buildings remain rock steady.",
        category="camera",
        seed=2006,
        difficulty="standard",
        target_metric="background_consistency",
        description="Zero camera drift test, background stability under moving foreground objects.",
    ),
    ValidationPrompt(
        id="camera_07",
        prompt="Handheld documentary camera following a traveler walking through a crowded Moroccan souk, subtle organic human sway.",
        category="camera",
        seed=2007,
        difficulty="challenging",
        target_metric="motion_smoothness",
        description="Natural organic handheld shake without high-frequency digital jitter.",
    ),
    ValidationPrompt(
        id="camera_08",
        prompt="Macro camera tracking forward along the spine of an antique leather-bound book, dust motes floating in sunbeams.",
        category="camera",
        seed=2008,
        difficulty="challenging",
        target_metric="subject_consistency",
        description="Shallow depth of field, slow forward translation, macro focus maintenance.",
    ),
    ValidationPrompt(
        id="camera_09",
        prompt="Sweeping aerial helicopter tracking shot over breaking ocean waves rolling into a sunset tropical beach.",
        category="camera",
        seed=2009,
        difficulty="standard",
        target_metric="aesthetic",
        description="Large-scale environmental perspective, shoreline parallax.",
    ),
    ValidationPrompt(
        id="camera_10",
        prompt="Reverse tracking shot gliding backwards ahead of a smiling golden retriever trotting happily through a green meadow.",
        category="camera",
        seed=2010,
        difficulty="standard",
        target_metric="subject_consistency",
        description="Subject centered while foreground ground plane streams rapidly backwards.",
    ),

    # 3. Counting & Precise Numerosity (10 prompts)
    ValidationPrompt(
        id="count_01",
        prompt="Exactly three bright red apples resting in a neat triangle on a rustic wooden kitchen table.",
        category="counting",
        seed=3001,
        difficulty="standard",
        target_metric="text_alignment",
        description="Verification of small integer counting (N=3), item permanence without ghosting.",
    ),
    ValidationPrompt(
        id="count_02",
        prompt="Five yellow sunflowers swaying gently in the wind, each flower head distinctly visible side-by-side.",
        category="counting",
        seed=3002,
        difficulty="challenging",
        target_metric="text_alignment",
        description="Count N=5, separation of identical adjacent objects during motion.",
    ),
    ValidationPrompt(
        id="count_03",
        prompt="A flock of exactly eight white seagulls flying in formation across a clear blue coastal sky.",
        category="counting",
        seed=3003,
        difficulty="stress_test",
        target_metric="text_alignment",
        description="Counting flying dynamic entities (N=8) without merging or splitting.",
    ),
    ValidationPrompt(
        id="count_04",
        prompt="Two playful ginger kittens wrestling over a single blue yarn ball on a carpet.",
        category="counting",
        seed=3004,
        difficulty="standard",
        target_metric="subject_consistency",
        description="Pair interaction (N=2), identity preservation during occlusion.",
    ),
    ValidationPrompt(
        id="count_05",
        prompt="Four transparent soap bubbles floating gently upward, catching iridescent rainbow reflections.",
        category="counting",
        seed=3005,
        difficulty="challenging",
        target_metric="text_alignment",
        description="Translucent discrete objects (N=4) with specular optical properties.",
    ),
    ValidationPrompt(
        id="count_06",
        prompt="A single silver spoon stirring a white porcelain cup of steaming black coffee.",
        category="counting",
        seed=3006,
        difficulty="standard",
        target_metric="text_alignment",
        description="Singular object focus (N=1), prevents spurious duplicate generation.",
    ),
    ValidationPrompt(
        id="count_07",
        prompt="Six colorful macarons displayed in a straight line inside a Parisian patisserie glass case.",
        category="counting",
        seed=3007,
        difficulty="challenging",
        target_metric="text_alignment",
        description="Discrete colored items (N=6) in structured linear arrangement.",
    ),
    ValidationPrompt(
        id="count_08",
        prompt="Three candles of varying heights burning brightly in a quiet dark room, wax slowly dripping.",
        category="counting",
        seed=3008,
        difficulty="standard",
        target_metric="text_alignment",
        description="Exact count N=3 with distinct height attributes and flickering flame dynamics.",
    ),
    ValidationPrompt(
        id="count_09",
        prompt="Two astronauts in white spacesuits tethered together doing a spacewalk outside the International Space Station.",
        category="counting",
        seed=3009,
        difficulty="challenging",
        target_metric="subject_consistency",
        description="Two interacting humans in high-frequency geometric suits against Earth backdrop.",
    ),
    ValidationPrompt(
        id="count_10",
        prompt="Seven colorful hot air balloons rising together over Cappadocia at dawn.",
        category="counting",
        seed=3010,
        difficulty="stress_test",
        target_metric="text_alignment",
        description="Multi-object aerial scene (N=7), vertical ascent tracking.",
    ),

    # 4. Text Legibility & Signage Rendering (10 prompts)
    ValidationPrompt(
        id="text_01",
        prompt="A glowing cyan neon sign mounted on brick reading 'WAN 2.1' with soft neon flicker and rain reflections.",
        category="text",
        seed=4001,
        difficulty="challenging",
        target_metric="text_alignment",
        description="Specific character glyph rendering ('WAN 2.1'), luminescence bloom.",
    ),
    ValidationPrompt(
        id="text_02",
        prompt="A wooden cafe chalkboard on a sidewalk with white chalk letters clearly spelling 'FRESH COFFEE' surrounded by small flowers.",
        category="text",
        seed=4002,
        difficulty="challenging",
        target_metric="text_alignment",
        description="All-caps chalkboard typography ('FRESH COFFEE'), textured background.",
    ),
    ValidationPrompt(
        id="text_03",
        prompt="A weathered wooden trail signpost with carved letters reading 'NORTH TRAIL 5 KM' pointing right in a foggy forest.",
        category="text",
        seed=4003,
        difficulty="challenging",
        target_metric="text_alignment",
        description="Multi-word signpost ('NORTH TRAIL 5 KM') in natural outdoor environment.",
    ),
    ValidationPrompt(
        id="text_04",
        prompt="An old vintage cinema marquee glowing in the evening displaying 'NOW PLAYING' in bold black capital letters.",
        category="text",
        seed=4004,
        difficulty="standard",
        target_metric="text_alignment",
        description="Classic marquee letterboard ('NOW PLAYING'), incandescent backlight.",
    ),
    ValidationPrompt(
        id="text_05",
        prompt="A delivery drone with bold yellow text 'VIDGEN AIR' printed on its white composite fuselage flying across the frame.",
        category="text",
        seed=4005,
        difficulty="stress_test",
        target_metric="text_alignment",
        description="Moving text on a 3D translating vehicle ('VIDGEN AIR'), perspective warping.",
    ),
    ValidationPrompt(
        id="text_06",
        prompt="A retro electronic LED ticker tape scrolling red letters: 'BREAKING NEWS' from right to left smoothly.",
        category="text",
        seed=4006,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Horizontal text translation and dot-matrix pixel rasterization.",
    ),
    ValidationPrompt(
        id="text_07",
        prompt="A barista pouring steamed milk into espresso creating a latte art heart, beside a ceramic cup printed with 'CAFE'.",
        category="text",
        seed=4007,
        difficulty="standard",
        target_metric="text_alignment",
        description="Small label text ('CAFE') alongside fluid latte art mechanics.",
    ),
    ValidationPrompt(
        id="text_08",
        prompt="A red stop sign at a snowy road intersection clearly showing the white letters 'STOP' dusted with fresh snow.",
        category="text",
        seed=4008,
        difficulty="standard",
        target_metric="text_alignment",
        description="Standardized iconic signage ('STOP'), partial weather occlusion.",
    ),
    ValidationPrompt(
        id="text_09",
        prompt="A futuristic holographic user interface projecting floating glowing blue text: 'SYSTEM ONLINE 100%'.",
        category="text",
        seed=4009,
        difficulty="challenging",
        target_metric="text_alignment",
        description="Semi-transparent holographic text rendering ('SYSTEM ONLINE 100%').",
    ),
    ValidationPrompt(
        id="text_10",
        prompt="A classic red London double-decker bus driving past Big Ben with destination sign reading 'PICCADILLY'.",
        category="text",
        seed=4010,
        difficulty="challenging",
        target_metric="text_alignment",
        description="Moving vehicle destination roller blind typography ('PICCADILLY').",
    ),

    # 5. People & Complex Human Kinematics (10 prompts)
    ValidationPrompt(
        id="people_01",
        prompt="An elderly artisan watchmaker with magnifying loupe examining intricate mechanical gears, smiling warmly in close-up.",
        category="people",
        seed=5001,
        difficulty="standard",
        target_metric="subject_consistency",
        description="Facial micro-expressions, skin wrinkles, hand stability, emotional realism.",
    ),
    ValidationPrompt(
        id="people_02",
        prompt="A master chef vigorously hand-pulling fresh wheat noodles in a cloud of flour, stretching and folding the dough.",
        category="people",
        seed=5002,
        difficulty="challenging",
        target_metric="dynamic_degree",
        description="Complex finger and arm dexterity, elastic deformable material interaction.",
    ),
    ValidationPrompt(
        id="people_03",
        prompt="A young girl blowing dandelion seeds towards the camera in golden hour sunlight, laughing with wind in her hair.",
        category="people",
        seed=5003,
        difficulty="standard",
        target_metric="aesthetic",
        description="Oral expiration motion, particle dispersal, joyful human expression.",
    ),
    ValidationPrompt(
        id="people_04",
        prompt="A flamenco dancer in an embroidered red dress stomping heels and snapping castanets in dramatic rhythmic motion.",
        category="people",
        seed=5004,
        difficulty="stress_test",
        target_metric="motion_smoothness",
        description="Rapid full-body kinetic articulation, fabric flow, sharp rhythm.",
    ),
    ValidationPrompt(
        id="people_05",
        prompt="A surgeon wearing surgical scrubs and sterile gloves carefully adjusting an overhead operating theater lamp.",
        category="people",
        seed=5005,
        difficulty="standard",
        target_metric="subject_consistency",
        description="Accurate human anatomy, 5-finger hand topology, medical setting.",
    ),
    ValidationPrompt(
        id="people_06",
        prompt="A crowd of diverse people walking across a rain-slicked crosswalk under colorful umbrellas at night.",
        category="people",
        seed=5006,
        difficulty="stress_test",
        target_metric="subject_consistency",
        description="Multi-agent pedestrian crowd simulation, independent gait cycles.",
    ),
    ValidationPrompt(
        id="people_07",
        prompt="A scientist in a clean laboratory peering through a high-powered microscope and writing notes with a fountain pen.",
        category="people",
        seed=5007,
        difficulty="standard",
        target_metric="subject_consistency",
        description="Subtle precise motor actions, posture maintenance.",
    ),
    ValidationPrompt(
        id="people_08",
        prompt="A hip-hop breakdancer performing a one-handed freeze spin on a polished floor in an urban underground studio.",
        category="people",
        seed=5008,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Acrobatic inversion, non-canonical body orientation, rotational physics.",
    ),
    ValidationPrompt(
        id="people_09",
        prompt="A father holding his infant baby high up in the air near a sunny window, baby giggling playfully.",
        category="people",
        seed=5009,
        difficulty="standard",
        target_metric="aesthetic",
        description="Parent-child physical contact, tender natural facial motion.",
    ),
    ValidationPrompt(
        id="people_10",
        prompt="An acoustic guitarist strumming intricate fingerstyle patterns on steel strings, fingers pressing on frets in close-up.",
        category="people",
        seed=5010,
        difficulty="challenging",
        target_metric="motion_smoothness",
        description="Fine motor hand articulation, string vibration.",
    ),

    # 6. Animals & Wildlife Locomotion (10 prompts)
    ValidationPrompt(
        id="animals_01",
        prompt="A sleek cheetah sprinting full speed across African savanna grassland, golden dust kicking up from powerful paws.",
        category="animals",
        seed=6001,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Quadruped galloping gait cycle, spine flexing, muscle definition.",
    ),
    ValidationPrompt(
        id="animals_02",
        prompt="A tiny ruby-throated hummingbird hovering in mid-air and sipping nectar from a vibrant fuchsia blossom.",
        category="animals",
        seed=6002,
        difficulty="challenging",
        target_metric="motion_smoothness",
        description="High-frequency wing flutter, stationary aerodynamic hover.",
    ),
    ValidationPrompt(
        id="animals_03",
        prompt="A translucent bioluminescent jellyfish pulsing gently through midnight deep ocean water, trailing glowing tentacles.",
        category="animals",
        seed=6003,
        difficulty="standard",
        target_metric="aesthetic",
        description="Soft-body hydrostatic contraction, bioluminescent emission.",
    ),
    ValidationPrompt(
        id="animals_04",
        prompt="A playful golden retriever puppy chasing a bouncing tennis ball across a lush green backyard lawn.",
        category="animals",
        seed=6004,
        difficulty="standard",
        target_metric="motion_smoothness",
        description="Canine bounding gait, playful momentum, bouncing ball interaction.",
    ),
    ValidationPrompt(
        id="animals_05",
        prompt="A magnificent humpback whale breaching out of calm Arctic ocean water and crashing back down in massive spray.",
        category="animals",
        seed=6005,
        difficulty="challenging",
        target_metric="dynamic_degree",
        description="Enormous scale inertia, water displacement, gravity descent.",
    ),
    ValidationPrompt(
        id="animals_06",
        prompt="A colorful chameleon sitting on a rainforest branch, eyes moving independently before extending tongue to catch a fly.",
        category="animals",
        seed=6006,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Independent ocular saccades, high-velocity projectile tongue mechanics.",
    ),
    ValidationPrompt(
        id="animals_07",
        prompt="A majestic bald eagle gliding through a rugged mountain gorge, banking into a thermal updraft.",
        category="animals",
        seed=6007,
        difficulty="standard",
        target_metric="motion_smoothness",
        description="Soaring flight aerodynamics, feather trim adjustments, mountain backdrop.",
    ),
    ValidationPrompt(
        id="animals_08",
        prompt="A red fox trotting through deep winter snow, suddenly pouncing headfirst to hunt beneath the frozen crust.",
        category="animals",
        seed=6008,
        difficulty="challenging",
        target_metric="dynamic_degree",
        description="Winter coat fur physics, parabolic pounce arc, snow penetration.",
    ),
    ValidationPrompt(
        id="animals_09",
        prompt="A clownfish swimming playfully through the waving tentacles of a sea anemone on a sunlit coral reef.",
        category="animals",
        seed=6009,
        difficulty="standard",
        target_metric="subject_consistency",
        description="Pectoral fin swimming, soft undulating invertebrate tendrils.",
    ),
    ValidationPrompt(
        id="animals_10",
        prompt="A mother brown bear leading her two small cubs across a rushing rocky river, water splashing around their legs.",
        category="animals",
        seed=6010,
        difficulty="challenging",
        target_metric="subject_consistency",
        description="Family group coordination, fur wetness rendering, turbulent shallow river.",
    ),

    # 7. Stylized Art, Cinematography & Rendering Aesthetics (10 prompts)
    ValidationPrompt(
        id="styles_01",
        prompt="Vintage 1970s Panavision 35mm film aesthetic, warm anamorphic lens flare, organic Kodak grain, a vintage red convertible on coastal highway.",
        category="styles",
        seed=7001,
        difficulty="standard",
        target_metric="aesthetic",
        description="Analog film characteristics, organic silver halide grain, optical flare.",
    ),
    ValidationPrompt(
        id="styles_02",
        prompt="Studio Ghibli anime cel animation of a cozy countryside kitchen with steaming tea kettle and sunlight streaming through cottage window.",
        category="styles",
        seed=7002,
        difficulty="standard",
        target_metric="aesthetic",
        description="Hand-painted gouache backgrounds, cel-shaded anime aesthetic, cozy atmospheric lighting.",
    ),
    ValidationPrompt(
        id="styles_03",
        prompt="Tactile stop-motion claymation of a whimsical green creature molding a tiny teacup from orange modeling clay, visible thumbprint textures.",
        category="styles",
        seed=7003,
        difficulty="challenging",
        target_metric="motion_smoothness",
        description="Step-frame tactile stop-motion (12 fps feel), clay surface imperfections.",
    ),
    ValidationPrompt(
        id="styles_04",
        prompt="Cyberpunk synthwave aesthetic with neon magenta and cyan wireframe grids, chrome sports car speeding down an endless digital highway under a retro sun.",
        category="styles",
        seed=7004,
        difficulty="standard",
        target_metric="aesthetic",
        description="Outrun/synthwave color palette, scanlines, glowing geometric grid horizon.",
    ),
    ValidationPrompt(
        id="styles_05",
        prompt="Dynamic watercolor painting coming to life, vivid ink bleeding and spreading across damp textured paper into blooming cherry blossom tree.",
        category="styles",
        seed=7005,
        difficulty="stress_test",
        target_metric="dynamic_degree",
        description="Generative fluid diffusion simulation, chromatic wash transitions.",
    ),
    ValidationPrompt(
        id="styles_06",
        prompt="1920s black and white silent film with authentic film gate weave, dust scratches, vignette, a comedic gentleman tipping his bowler hat.",
        category="styles",
        seed=7006,
        difficulty="standard",
        target_metric="aesthetic",
        description="Monochrome archival celluloid degradation, shutter flicker, high contrast.",
    ),
    ValidationPrompt(
        id="styles_07",
        prompt="Origami papercraft world where a folded white paper swan flaps paper wings and glides along geometric folded blue paper waves.",
        category="styles",
        seed=7007,
        difficulty="challenging",
        target_metric="motion_smoothness",
        description="Creased paper folding physics, crisp geometric planar surfaces.",
    ),
    ValidationPrompt(
        id="styles_08",
        prompt="Impressionist oil painting in the style of Van Gogh, thick impasto swirling brushstrokes depicting a starry night sky over a French village in motion.",
        category="styles",
        seed=7008,
        difficulty="challenging",
        target_metric="aesthetic",
        description="Turbulent brushstroke animation, vibrant contrasting oil pigments.",
    ),
    ValidationPrompt(
        id="styles_09",
        prompt="1980s VHS magnetic tape recording with tracking static lines, chromatic aberration bleed, a teenager playing an arcade machine.",
        category="styles",
        seed=7009,
        difficulty="standard",
        target_metric="aesthetic",
        description="NTSC composite video artifacts, phosphor glow, magnetic tape distortion.",
    ),
    ValidationPrompt(
        id="styles_10",
        prompt="Hyper-detailed cinematic IMAX 70mm nature documentary footage of glowing volcanic lava flows meeting the cold ocean with billowing steam.",
        category="styles",
        seed=7010,
        difficulty="standard",
        target_metric="aesthetic",
        description="Ultra-high fidelity dynamic range, basalt textures, glowing incandescent magma.",
    ),
]


def get_all_validation_prompts() -> List[ValidationPrompt]:
    """Return all 70 validation benchmark prompts."""
    return list(VALIDATION_PROMPTS_CATALOG)


def get_prompts_by_category(category: str) -> List[ValidationPrompt]:
    """Filter validation prompts by specific category."""
    return [p for p in VALIDATION_PROMPTS_CATALOG if p.category == category]


def get_sample_grid_prompts(count_per_category: int = 1) -> List[ValidationPrompt]:
    """Return a balanced subset representing each category for side-by-side comparison grids."""
    categories = ['motion', 'camera', 'counting', 'text', 'people', 'animals', 'styles']
    selected: List[ValidationPrompt] = []
    for cat in categories:
        cat_prompts = get_prompts_by_category(cat)
        selected.extend(cat_prompts[:count_per_category])
    return selected


def export_prompts_as_dict() -> List[Dict[str, Any]]:
    """Export validation prompts as JSON-serializable dictionaries."""
    return [asdict(p) for p in VALIDATION_PROMPTS_CATALOG]
