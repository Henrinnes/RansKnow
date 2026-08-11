const fs = require('fs');
const { runRuleBased, tfidfVector, runMulticlassTask, runMultilabelTask, tokenize } = require('./assets/ka_engine.js');

const ruleData = JSON.parse(fs.readFileSync(__dirname + '/../outputs/rulebased_patterns.json'));
const modelData = JSON.parse(fs.readFileSync(__dirname + '/../outputs/demo_models.json'));

console.log('=== Reference test: TF-IDF + LogReg dominant_tactic ===\n');

const tests = [
  "The attackers used LockBit ransomware to encrypt files across the network after gaining initial access via phishing email.",
  "This talk covers lateral movement techniques using PsExec and WMI to spread across the domain.",
  "A completely unrelated video about cooking pasta and Italian cuisine recipes.",
];

const expected = [
  { Initial_Access: 0.9974, Impact: 0.0020, Exfiltration: 0.0002 },
  { Discovery: 0.3351, Initial_Access: 0.2068, Impact: 0.1666 },
  { Impact: 0.9881, Persistence: 0.0060, Initial_Access: 0.0020 },
];

let allPass = true;

tests.forEach((s, i) => {
  const vec = tfidfVector(s, modelData.vocabulary, modelData.idf, modelData.vocabulary && Object.keys(modelData.vocabulary).length ? modelData.idf.length : 3000);
  const probs = runMulticlassTask(modelData.tasks.dominant_tactic, 'logreg', vec);
  const top3 = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3);

  console.log(`[${i}] "${s.slice(0, 50)}..."`);
  console.log('  JS top3:', top3.map(([c, p]) => `${c}=${p.toFixed(4)}`).join(', '));
  console.log('  PY top3:', Object.entries(expected[i]).map(([c, p]) => `${c}=${p.toFixed(4)}`).join(', '));

  for (const [cls, pyVal] of Object.entries(expected[i])) {
    const jsVal = probs[cls];
    const diff = Math.abs(jsVal - pyVal);
    const ok = diff < 0.001;
    if (!ok) allPass = false;
    console.log(`    ${cls}: js=${jsVal.toFixed(4)} py=${pyVal.toFixed(4)} diff=${diff.toFixed(5)} ${ok ? 'OK' : 'MISMATCH'}`);
  }
  console.log();
});

console.log('=== Rule-based extraction sanity check ===\n');
const ruleTest = "The LockBit ransomware gang used Cobalt Strike and PsExec for lateral movement, then exfiltrated data via rclone before encrypting files on ESXi hosts.";
const result = runRuleBased(ruleTest, ruleData);
console.log('families:', result.families);
console.log('dominantTactic:', result.dominantTactic);
console.log('platformSignal:', result.platformSignal);
console.log('toolList:', result.toolList);

console.log('\n=== Ambiguous alias context-gating check ("play" without context) ===');
const noContext = "Let's play the video and see what happens next in this segment.";
const r2 = runRuleBased(noContext, ruleData);
console.log('families (should be empty, no ransomware context near "play"):', r2.families);

console.log('\n=== Ambiguous alias context-gating check ("play" WITH context) ===');
const withContext = "The Play ransomware group encrypted victim files and demanded payment, according to this cyber threat intelligence report on the extortion group.";
const r3 = runRuleBased(withContext, ruleData);
console.log('families (should include Play):', r3.families);

console.log('\n' + (allPass ? 'ALL NUMERIC CHECKS PASSED' : 'SOME CHECKS FAILED -- see MISMATCH above'));
