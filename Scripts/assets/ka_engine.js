// RansKnow inference engine -- faithful JS port of knowledge_agent.py's
// regex extraction and the exported sklearn TF-IDF + linear model
// pipelines. See Scripts/export_rulebased_patterns.py and
// Scripts/export_demo_models.py for how RULE_DATA / MODEL_DATA are produced.

function normText(text) {
  return text.toLowerCase().replace(/\s+/g, ' ');
}

// ---- Rule-based Knowledge Agent extraction ----

function countMatches(text, patterns) {
  let total = 0;
  for (const p of patterns) {
    const re = new RegExp(p, 'gi');
    const m = text.match(re);
    if (m) total += m.length;
  }
  return total;
}

function hasContext(text, pos, contextTerms, windowChars) {
  const start = Math.max(0, pos - windowChars);
  const end = Math.min(text.length, pos + windowChars);
  const span = text.slice(start, end);
  return contextTerms.some(t => span.includes(t));
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function findFamilies(text, ruleData) {
  const found = new Set();
  for (const [alias, canon] of Object.entries(ruleData.alias_map)) {
    const pat = escapeRegex(alias).replace(/\\ /g, '\\s+');
    const re = new RegExp('\\b' + pat + '\\b', 'gi');
    if (!ruleData.ambiguous_aliases.includes(alias)) {
      if (re.test(text)) found.add(canon);
      continue;
    }
    // ambiguous alias: require a disambiguating context term nearby
    let m;
    const re2 = new RegExp('\\b' + pat + '\\b', 'gi');
    while ((m = re2.exec(text)) !== null) {
      if (hasContext(text, m.index, ruleData.context_terms, ruleData.context_window_chars)) {
        found.add(canon);
        break;
      }
      if (m.index === re2.lastIndex) re2.lastIndex++; // avoid infinite loop on zero-width match
    }
  }
  return Array.from(found).sort();
}

function runRuleBased(rawText, ruleData) {
  const text = normText(rawText);

  const families = findFamilies(text, ruleData);

  const tacticCounts = {};
  for (const [t, pats] of Object.entries(ruleData.tactics)) tacticCounts[t] = countMatches(text, pats);
  const toolCounts = {};
  for (const [t, pats] of Object.entries(ruleData.tools)) toolCounts[t] = countMatches(text, pats);
  const platCounts = {};
  for (const [p, pats] of Object.entries(ruleData.platforms)) platCounts[p] = countMatches(text, pats);

  // exact tie-break parity with Python max(dict, key=dict.get): first
  // key (in insertion/object order) wins ties.
  function pyStyleArgmax(counts) {
    let best = null, bestVal = -Infinity, any = false;
    for (const [k, v] of Object.entries(counts)) {
      any = any || v > 0;
      if (v > bestVal) { bestVal = v; best = k; }
    }
    return any ? best : null;
  }

  return {
    families,
    tacticCounts,
    toolCounts,
    platCounts,
    dominantTactic: pyStyleArgmax(tacticCounts) || 'None',
    platformSignal: pyStyleArgmax(platCounts) || 'None',
    toolList: Object.entries(toolCounts).filter(([, v]) => v > 0).map(([k]) => k),
  };
}

// ---- TF-IDF vectorization (sklearn-compatible) ----

// sklearn default token pattern: (?u)\b\w\w+\b  (word chars, length >= 2)
function tokenize(text) {
  const matches = text.toLowerCase().match(/\b\w{2,}\b/g) || [];
  return matches;
}

function ngrams(tokens, n) {
  const out = [];
  for (let i = 0; i + n <= tokens.length; i++) {
    out.push(tokens.slice(i, i + n).join(' '));
  }
  return out;
}

function tfidfVector(rawText, vocabulary, idf, vocabSize) {
  const tokens = tokenize(rawText);
  const terms = ngrams(tokens, 1).concat(ngrams(tokens, 2));

  const counts = new Map();
  for (const t of terms) {
    if (Object.prototype.hasOwnProperty.call(vocabulary, t)) {
      counts.set(t, (counts.get(t) || 0) + 1);
    }
  }

  const vec = new Float64Array(vocabSize);
  for (const [term, count] of counts) {
    const idx = vocabulary[term];
    vec[idx] = count * idf[idx];
  }

  // L2 normalize
  let normSq = 0;
  for (let i = 0; i < vocabSize; i++) normSq += vec[i] * vec[i];
  const norm = Math.sqrt(normSq);
  if (norm > 0) for (let i = 0; i < vocabSize; i++) vec[i] /= norm;

  return vec;
}

// ---- Linear model scoring ----

function scaleVector(vec, scale) {
  const out = new Float64Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / scale[i];
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// Score a single fitted binary/multiclass pipeline export:
// { scale, coef (n_classes x n_features), intercept (n_classes), classes }
function scorePipeline(exported, tfidfVec) {
  const scaled = scaleVector(tfidfVec, exported.scale);
  const scores = exported.coef.map((row, i) => dot(row, scaled) + exported.intercept[i]);
  if (exported.classes.length === 2 && exported.coef.length === 1) {
    const p1 = sigmoid(scores[0]);
    return { [exported.classes[0]]: 1 - p1, [exported.classes[1]]: p1 };
  }
  const probs = softmax(scores);
  const out = {};
  exported.classes.forEach((c, i) => { out[c] = probs[i]; });
  return out;
}

// LinearSVC has no predict_proba -- raw decision margins only.
function scoreSvmPipeline(exported, tfidfVec) {
  const scaled = scaleVector(tfidfVec, exported.scale);
  const scores = exported.coef.map((row, i) => dot(row, scaled) + exported.intercept[i]);
  const out = {};
  exported.classes.forEach((c, i) => { out[c] = scores[i]; });
  return out;
}

function runMulticlassTask(taskExport, modelName, tfidfVec) {
  const exported = taskExport.models[modelName];
  return modelName === 'svm' ? scoreSvmPipeline(exported, tfidfVec) : scorePipeline(exported, tfidfVec);
}

function runMultilabelTask(taskExport, modelName, tfidfVec) {
  const perClass = taskExport.models[modelName];
  const out = {};
  for (const cls of perClass) {
    if ('constant' in cls) {
      out[cls.class_name] = cls.constant === 1 ? 1.0 : 0.0;
      continue;
    }
    if (modelName === 'svm') {
      const scaled = scaleVector(tfidfVec, cls.scale);
      const score = dot(cls.coef[0], scaled) + cls.intercept[0];
      out[cls.class_name] = score; // margin, not probability
    } else {
      const scaled = scaleVector(tfidfVec, cls.scale);
      const score = dot(cls.coef[0], scaled) + cls.intercept[0];
      out[cls.class_name] = sigmoid(score);
    }
  }
  return out;
}

if (typeof module !== 'undefined') {
  module.exports = { normText, runRuleBased, tfidfVector, runMulticlassTask, runMultilabelTask, tokenize };
}
