/**
 * Test runner script for dashboard engine tests
 */

import { runDashboardEngineTests } from '../src/utils/__tests__/dashboardEngine.test';

async function main() {
  console.log('--- Running Vidgen Dashboard Engine Unit Tests ---');
  const { passed, failed, results } = await runDashboardEngineTests();

  for (const r of results) {
    if (r.passed) {
      console.log(`  ✓ ${r.name}`);
    } else {
      console.error(`  ✗ ${r.name} - ${r.message}`);
    }
  }

  console.log(`\nTest Summary: ${passed} passed, ${failed} failed.`);

  if (failed > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

main().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});
