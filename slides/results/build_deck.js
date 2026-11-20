#!/usr/bin/env node
/**
 * Build slides/results/results_deck.pptx from deck_data.json.
 * One SCRIPT array drives pptx notes, speaker_notes.md, and narration_script.txt.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname);
const DATA = JSON.parse(fs.readFileSync(path.join(ROOT, "deck_data.json"), "utf8"));
const OUT = path.join(ROOT, "results_deck.pptx");
const SKILL_APPLY =
  process.env.PPTX_APPLY_THEME ||
  path.join(
    process.env.HOME,
    ".claude/skills/synced/ba69b7e1-f825-4e77-8caf-b28481a9a29e_66307069-1c04-4838-8c33-833918066dc8/pptx/scripts/apply_theme.js"
  );

const THEME = {
  name: "Interp Charcoal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A1A1A",
    lt1: "FFFFFF",
    dk2: "36454F",
    lt2: "F4F6F7",
    accent1: "0D7377",
    accent2: "E07A3D",
    accent3: "6B3A6B",
    accent4: "C0392B",
    accent5: "2C3E50",
    accent6: "7F8C8D",
    hlink: "0D7377",
    folHlink: "6B3A6B",
  },
};

const COL = {
  charcoal: "36454F",
  ink: "1A1A1A",
  white: "FFFFFF",
  light: "F4F6F7",
  card: "EEF2F4",
  teal: "0D7377",
  saffron: "E07A3D",
  plum: "6B3A6B",
  fail: "C0392B",
  pass: "1E8449",
  muted: "6B7B85",
  softFail: "F5D5D1",
  softPass: "D5EDE0",
  softFirst: "F5E6C8",
};

function fmt(x, d = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(d);
}

function speakable(n, d = 2) {
  const s = fmt(n, d);
  if (s.startsWith("-")) return "minus " + s.slice(1);
  return s;
}

/* -------------------- Spoken script (single source) -------------------- */

const AP_X = fmt(DATA.cost.ap_over_attrp, 0);
const EAP_X = fmt(DATA.cost.eap_over_attrp, 1);
const MEAN_AP = fmt(DATA.cost.mean_t_ap, 0);
const MEAN_ATTR = fmt(DATA.cost.mean_t_attr, 1);
const H1_N = DATA.h1.n_cells;
const H1_FAIL_N = Array.isArray(DATA.h1.failing) ? DATA.h1.failing.length : Number(DATA.h1.failing);
const H1_PASS = H1_N - H1_FAIL_N;
const H1_THR = fmt(DATA.h1.threshold, 2);
const MLP0_AP = speakable(DATA.mlp_layers.gpt2_ioi.ap[0], 2);
const MLP0_ATTR = speakable(DATA.mlp_layers.gpt2_ioi.attr[0], 2);
const EAP_MLP_BEFORE = speakable(DATA.eap_examples.gpt2_ioi.mlp_r, 2);
const EAP_MLP_AFTER = speakable(DATA.eap_examples.gpt2_ioi.eap_mlp_r, 2);
const EAP_RHO_BEFORE = speakable(DATA.eap_examples.gpt2_ioi.rho, 2);
const EAP_RHO_AFTER = speakable(DATA.eap_examples.gpt2_ioi.eap_rho, 2);
const HEAD_DROP = speakable(-DATA.h5.head_rho_gain_mean, 2);
const H3_DIFF = speakable(DATA.h3.pooled_mean_diff, 3);
const H3_LO = speakable(DATA.h3.ci95[0], 3);
const H3_HI = speakable(DATA.h3.ci95[1], 3);
const NOISE_RHO = speakable(DATA.noising_mean_rho, 2);
const FAIL_ACC_LO = fmt(DATA.hindi_fail_acc.min * 100, 0);
const FAIL_ACC_HI = fmt(DATA.hindi_fail_acc.max * 100, 0);
const SWEEP = DATA.h5.step_sweep_seed0;
const SWEEP_RHO = ["5", "10", "20"].map((k) => speakable(SWEEP[k].mean_head_rho, 2)).join(", ");
const BLOOM_MLP = speakable(Math.min(...Object.values(DATA.bloom_ioi_mlp_r)), 3);
const MISS0 = DATA.h1.bars.filter((b) => !b.pass_h1);
const MISS_A = speakable(MISS0[0].rho, 2);
const MISS_B = speakable(MISS0[1].rho, 2);

const SCRIPT = [
  {
    id: 1,
    title: "Title",
    say: [
      `Activation patching finds which attention heads and MLP layers matter for a behaviour. Exact, and expensive. Attribution patching is the cheap first-order shortcut people use when models get bigger.`,
      `I measured when the cheap method matches the expensive one, across tasks, models, and English and Hindi. Short answer: for heads, almost always. For MLPs, not always, and the failure is usually one early layer. Hindi only where the model can do the task.`,
    ].join("\n\n"),
    ask: [
      `Is this a new method? No. I measure when an existing cheap method matches the expensive one, across languages.`,
    ],
  },
  {
    id: 2,
    title: "Trade-off",
    say: [
      `On the left, activation patching: copy one component from a clean run into a corrupted run. Exact, slow. On the right, attribution patching: a Taylor approximation, one forward and one backward, every component ranked at once.`,
      `On this grid the median wall-time ratio is about ${AP_X}: roughly ${MEAN_AP} seconds against ${MEAN_ATTR} seconds per run on a 2080 Ti. When do they agree?`,
    ].join("\n\n"),
    ask: [
      `Why not always use activation patching? At one and a half billion parameters and many cells, it is the bottleneck.`,
      `The ${AP_X} is median wall time over ${DATA.cost.n_runs_timed} core denoising runs, not a FLOP count.`,
    ],
  },
  {
    id: 3,
    title: "What I ran",
    say: [
      `Nine models under two billion parameters: GPT-2, Pythia, Qwen, Llama, BLOOM. Four tasks in English and Hindi: indirect object, capitals, agreement, induction. A cell is one model on one task in one language.`,
      `Before I patch, I gate on competence: clean accuracy at least point eight, and a clear logit difference. Sixty-six cells gated, forty-four runnable. Then ${DATA.meta.n_runs} runs, five seeds, zero failures.`,
    ].join("\n\n"),
    ask: [
      `Why five seeds? They resample prompts, not weights.`,
      `Gemma three was in the proposal; tooling could not load it, so BLOOM replaced it before any grid result.`,
    ],
  },
  {
    id: 4,
    title: "Hindi gate",
    say: [
      `Before I could compare anything, most Hindi tasks dropped out. English is green everywhere. Hindi indirect-object fails in all six models I gated, Devanagari and romanised. Capitals fails in five of six; only Llama gets it, and only on the first token. Failing accuracy is ${FAIL_ACC_LO} to ${FAIL_ACC_HI} percent, so there is nothing to localise.`,
      `What remains is BLOOM agreement, plus induction on BLOOM and Llama. I reviewed the Hindi templates myself; one place-word fix, re-gate unchanged.`,
    ].join("\n\n"),
    ask: [
      `Why so little Hindi? That is the finding. You cannot localise a behaviour the model does not have.`,
    ],
  },
  {
    id: 5,
    title: "Heads",
    say: [
      `Where the gate passes, do the methods agree on heads? Each bar is the Spearman correlation of head rankings. The threshold of ${H1_THR} was preregistered.`,
      `${H1_PASS} of ${H1_N} cells clear it. The two misses, in red, are Qwen English agreement at ${MISS_A} and ${MISS_B}. Still high, just under the bar. Noising averages about ${NOISE_RHO}. Is attribution worse in Hindi?`,
    ].join("\n\n"),
    ask: [
      `Is ${H1_THR} magic? No; it was fixed in the preregistration. Do not lean on restricted correlation over large-effect heads: higher than plain correlation in only eleven of twenty-two cells.`,
    ],
  },
  {
    id: 6,
    title: "Hindi vs English",
    say: [
      `The only full-gate English–Hindi pair is BLOOM agreement, three sizes. The bars are almost the same. Equivalence within plus or minus point zero five: pooled difference plus ${H3_DIFF}, ninety-five percent CI from ${H3_LO} to ${H3_HI}.`,
      `Caveat out loud: three pairs, one family, one task. I do not claim this for Hindi IOI. Heads look fine. What about MLPs?`,
    ].join("\n\n"),
    ask: [
      `Is BLOOM special? It is the only family with usable Hindi agreement. That is a limit, not a virtue.`,
    ],
  },
  {
    id: 7,
    title: "MLP layer 0",
    say: [
      `I predicted MLP anti-correlation in most cells. Rejected: only two of twenty-four. The GPT-2 IOI failure is real, and it is almost entirely layer zero: activation ${MLP0_AP}, attribution ${MLP0_ATTR}. From layer one onward the lines sit on top of each other.`,
      `Same pattern in Qwen point five B and both Pythias; that part is exploratory. BLOOM MLPs agree. Can integrated gradients fix layer zero without changing the head ranking?`,
    ].join("\n\n"),
    ask: [
      `Why layer zero? Early nonlinearities; the linear approximation is worst there.`,
    ],
  },
  {
    id: 8,
    title: "Integrated gradients",
    say: [
      `On GPT-2 IOI, integrated gradients fixes layer zero: MLP agreement goes from ${EAP_MLP_BEFORE} to ${EAP_MLP_AFTER}. But head ranking drops from ${EAP_RHO_BEFORE} to ${EAP_RHO_AFTER}. That trade is the point.`,
      `Across the grid, mean head correlation drops by ${HEAD_DROP}, and the MLP gain is indistinguishable from zero. Cost about ${EAP_X} times attribution patching. More steps do not help.`,
    ].join("\n\n"),
    ask: [
      `Alpha zero reduces to attribution patching; checked on GPT-2. The head drop is from integrating along the path.`,
      `Step sweep on seed 0: mean head correlation ${SWEEP_RHO} at five, ten, twenty steps.`,
    ],
  },
  {
    id: 9,
    title: "Takeaway",
    say: [
      `Three rules. Use attribution patching for heads: ${H1_PASS} of ${H1_N}, about ${AP_X} times cheaper. Check early MLPs with real activation patching; layer zero can flip sign. Gate the task before you localise.`,
      `Limits: Hindi evidence is narrow, mostly BLOOM agreement. Gemma was dropped for tooling. Seeds resample prompts. I reviewed the templates myself. Thank you.`,
    ].join("\n\n"),
    ask: [
      `What next? A larger Hindi-capable model with single-token agreement, or hand-hooked Gemma once tooling catches up.`,
    ],
  },
  {
    id: 10,
    title: "Backup: gate counts",
    say: `Backup. Full gate breakdown by task and language, if asked.`,
    ask: [],
  },
  {
    id: 11,
    title: "Backup: noising",
    say: `Backup. Secondary direction from the preregistration. Mean head correlation under noising is ${NOISE_RHO}, slightly lower than denoising, same qualitative story.`,
    ask: [],
  },
  {
    id: 12,
    title: "Backup: amendments and fertility",
    say: [
      `Backup for design-change questions and the fertility hypothesis.`,
      `Amendment 2 was written during the gate phase, before any grid result existed. Do not claim more than that: the gate-rule changes came after the first gate numbers, and the noising sign fix was found by a smoke test that compared the two methods when no study results existed.`,
      `Fertility and clean logit difference correlate with agreement at Spearman ${speakable(DATA.h4.rho_vs_tokens_per_word.rho, 2)} and ${speakable(DATA.h4.rho_vs_clean_ld.rho, 2)}. Observational only.`,
    ].join("\n\n"),
    ask: [],
  },
];

function notesFor(id) {
  const e = SCRIPT.find((x) => x.id === id);
  if (!e) return "";
  const parts = [e.say];
  if (e.ask && e.ask.length) {
    parts.push("If asked:\n" + e.ask.map((q) => "• " + q).join("\n"));
  }
  return parts.join("\n\n");
}

function writeSpeakerNotesMd() {
  const lines = [
    `# Speaker notes · results deck (≈10 min live)`,
    ``,
    `Spoken text is cut for a ~10 minute live talk. Aim under a minute per main slide; leave "If asked" for questions.`,
    `Spoken text is filled from the same SCRIPT that builds the deck; numbers come from deck_data.json.`,
    ``,
  ];
  for (const e of SCRIPT.filter((x) => x.id <= 9)) {
    lines.push(`## ${e.id}. ${e.title}`);
    lines.push(e.say);
    if (e.ask && e.ask.length) {
      lines.push(``);
      lines.push(`**If asked**`);
      for (const q of e.ask) lines.push(`- ${q}`);
    }
    lines.push(``);
  }
  lines.push(`## Backup`);
  lines.push(
    `Gate table · noising mean ${NOISE_RHO} · Amendment 2 (gate phase, before any grid result) · fertility observational (Spearman ${speakable(DATA.h4.rho_vs_tokens_per_word.rho, 2)} with tokens/word, ${speakable(DATA.h4.rho_vs_clean_ld.rho, 2)} with clean LD).`
  );
  lines.push(``);
  fs.writeFileSync(path.join(ROOT, "speaker_notes.md"), lines.join("\n"));
}

function writeNarrationScript() {
  const parts = [];
  for (const e of SCRIPT.filter((x) => x.id <= 9)) {
    parts.push(e.say);
  }
  const text = parts.join("\n\n");
  fs.writeFileSync(path.join(ROOT, "narration_script.txt"), text + "\n");
  const words = text.trim().split(/\s+/).length;
  console.log("narration words:", words);
}

async function main() {
  const pres = new pptxgen();
  pres.defineLayout({ name: "LAYOUT_16x9", width: 10, height: 5.625 });
  pres.layout = "LAYOUT_16x9";
  pres.author = DATA.meta.author;
  pres.title = "When can attribution patching stand in for activation patching?";
  pres.subject = "AttrP vs AP · English/Hindi multi-cell study";
  pres.company = "IIIT Hyderabad";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: COL.charcoal },
    objects: [
      {
        placeholder: {
          options: {
            name: "title",
            type: "title",
            x: 0.6,
            y: 1.6,
            w: 8.8,
            h: 1.6,
            color: COL.white,
            fontSize: 32,
            bold: true,
            align: "left",
            valign: "middle",
            margin: 0,
          },
        },
      },
    ],
  });

  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: COL.white },
    objects: [
      {
        placeholder: {
          options: {
            name: "title",
            type: "title",
            x: 0.5,
            y: 0.28,
            w: 9.0,
            h: 0.7,
            color: COL.ink,
            fontSize: 24,
            bold: true,
            margin: 0,
          },
        },
      },
      {
        text: {
          text: "AttrP vs AP · IIIT Hyderabad",
          options: {
            x: 0.5,
            y: 5.28,
            w: 7.5,
            h: 0.25,
            fontSize: 10,
            color: COL.muted,
            margin: 0,
          },
        },
      },
    ],
    slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, color: COL.muted, fontSize: 10 },
  });

  pres.defineSlideMaster({
    title: "CLOSE_DARK",
    background: { color: COL.charcoal },
    objects: [
      {
        placeholder: {
          options: {
            name: "title",
            type: "title",
            x: 0.5,
            y: 0.35,
            w: 9.0,
            h: 0.7,
            color: COL.white,
            fontSize: 26,
            bold: true,
            margin: 0,
          },
        },
      },
    ],
    slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, color: "A0A8B0", fontSize: 10 },
  });

  pres.addSection({ title: "Opening" });
  slide1(pres, C);
  slide2(pres, C);
  slide3(pres, C);

  pres.addSection({ title: "Findings" });
  slide4(pres, C);
  slide5(pres, C);
  slide6(pres, C);
  slide7(pres, C);
  slide8(pres, C);

  pres.addSection({ title: "Close" });
  slide9(pres, C);

  pres.addSection({ title: "Backup" });
  backup1(pres, C);
  backup2(pres, C);
  backup3(pres, C);

  await pres.writeFile({ fileName: OUT });

  try {
    if (!module.paths.includes(path.join(ROOT, "node_modules"))) {
      module.paths.unshift(path.join(ROOT, "node_modules"));
    }
    const { applyTheme } = require(SKILL_APPLY);
    await applyTheme(OUT, THEME);
    console.log("theme applied");
  } catch (e) {
    console.warn("applyTheme skipped:", e.message);
    console.warn("Hint: NODE_PATH=" + path.join(ROOT, "node_modules") + " node build_deck.js");
  }

  writeSpeakerNotesMd();
  writeNarrationScript();
  console.log("wrote", OUT);
}

/* -------------------- Slides -------------------- */

function slide1(pres, C) {
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addText("When can attribution patching\nstand in for activation patching?", {
    placeholder: "title",
  });
  s.addText("English and Hindi · nine models · a competence gate before every comparison", {
    x: 0.6,
    y: 3.4,
    w: 8.8,
    h: 0.4,
    fontSize: 16,
    color: "CAD5DC",
    margin: 0,
    isTextBox: true,
  });
  s.addText(`${DATA.meta.author}  ·  ${DATA.meta.course}`, {
    x: 0.6,
    y: 4.7,
    w: 8.8,
    h: 0.3,
    fontSize: 13,
    color: "A0A8B0",
    margin: 0,
    isTextBox: true,
  });
  [
    [COL.teal, "Activation"],
    [COL.saffron, "Attribution"],
    [COL.plum, "EAP-IG"],
  ].forEach(([c, label], i) => {
    const x = 0.6 + i * 1.7;
    s.addShape(pres.shapes.OVAL, { x, y: 4.05, w: 0.22, h: 0.22, fill: { color: c } });
    s.addText(label, {
      x: x + 0.3,
      y: 4.02,
      w: 1.3,
      h: 0.28,
      fontSize: 12,
      color: "CAD5DC",
      margin: 0,
      isTextBox: true,
    });
  });
  s.addNotes(notesFor(1));
}

function slide2(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  s.addText(`Activation patching is exact, but about ${AP_X} times slower`, {
    placeholder: "title",
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5,
    y: 1.15,
    w: 4.3,
    h: 2.7,
    fill: { color: COL.card },
    rectRadius: 0.1,
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 5.2,
    y: 1.15,
    w: 4.3,
    h: 2.7,
    fill: { color: COL.card },
    rectRadius: 0.1,
  });
  s.addShape(pres.shapes.OVAL, { x: 0.75, y: 1.35, w: 0.28, h: 0.28, fill: { color: COL.teal } });
  s.addShape(pres.shapes.OVAL, { x: 5.45, y: 1.35, w: 0.28, h: 0.28, fill: { color: COL.saffron } });
  s.addText("Activation patching", {
    x: 1.15,
    y: 1.32,
    w: 3.4,
    h: 0.35,
    fontSize: 16,
    bold: true,
    color: COL.ink,
    margin: 0,
    isTextBox: true,
  });
  s.addText("Attribution patching", {
    x: 5.85,
    y: 1.32,
    w: 3.4,
    h: 0.35,
    fontSize: 16,
    bold: true,
    color: COL.ink,
    margin: 0,
    isTextBox: true,
  });
  s.addText(
    [
      { text: "Exact causal test", options: { bold: true, breakLine: true } },
      { text: "One forward pass per head or MLP", options: { breakLine: true } },
      { text: "Cost grows with components and prompts", options: { breakLine: true } },
      { text: `About ${MEAN_AP} s per cell on a 2080 Ti`, options: {} },
    ],
    { x: 0.75, y: 1.85, w: 3.8, h: 1.7, fontSize: 14, color: COL.ink, paraSpaceAfter: 6, margin: 0, isTextBox: true }
  );
  s.addText(
    [
      { text: "First-order gradient shortcut", options: { bold: true, breakLine: true } },
      { text: "One forward pass and one backward pass", options: { breakLine: true } },
      { text: "Ranks every component at once", options: { breakLine: true } },
      { text: `About ${MEAN_ATTR} s per cell, roughly ${AP_X} times faster`, options: {} },
    ],
    { x: 5.45, y: 1.85, w: 3.8, h: 1.7, fontSize: 14, color: COL.ink, paraSpaceAfter: 6, margin: 0, isTextBox: true }
  );

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5,
    y: 4.1,
    w: 9.0,
    h: 0.85,
    fill: { color: COL.charcoal },
    rectRadius: 0.08,
  });
  s.addText(
    [
      { text: `~${AP_X}×  `, options: { bold: true, fontSize: 28, color: COL.saffron } },
      {
        text: "wall time of activation vs attribution patching on this grid. When do they agree?",
        options: { fontSize: 15, color: COL.white },
      },
    ],
    { x: 0.75, y: 4.25, w: 8.5, h: 0.55, margin: 0, valign: "middle", isTextBox: true }
  );

  s.addNotes(notesFor(2));
}

function slide3(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  s.addText("I check that the model can do the task before patching it", {
    placeholder: "title",
  });

  const steps = [
    { n: "1", t: "9 models", d: "GPT-2, Pythia, Qwen,\nLlama, BLOOM (≤2B)" },
    { n: "2", t: "4 tasks × en/hi", d: "IOI, capitals,\nagreement, induction" },
    { n: "3", t: "Competence gate", d: `${DATA.meta.n_gate_cells} cells →\n${DATA.meta.n_runnable} runnable` },
    { n: "4", t: `${DATA.meta.n_runs} runs`, d: "AP + AttrP + EAP-IG\n2 dirs × 5 seeds" },
  ];
  steps.forEach((st, i) => {
    const x = 0.5 + i * 2.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y: 1.2,
      w: 2.15,
      h: 2.5,
      fill: { color: COL.card },
      rectRadius: 0.1,
    });
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.75,
      y: 1.4,
      w: 0.55,
      h: 0.55,
      fill: { color: COL.charcoal },
    });
    s.addText(st.n, {
      x: x + 0.75,
      y: 1.48,
      w: 0.55,
      h: 0.4,
      fontSize: 18,
      bold: true,
      color: COL.white,
      align: "center",
      margin: 0,
      isTextBox: true,
    });
    s.addText(st.t, {
      x: x + 0.1,
      y: 2.15,
      w: 1.95,
      h: 0.55,
      fontSize: 15,
      bold: true,
      color: COL.ink,
      align: "center",
      margin: 0,
      isTextBox: true,
    });
    s.addText(st.d, {
      x: x + 0.1,
      y: 2.75,
      w: 1.95,
      h: 0.75,
      fontSize: 12,
      color: COL.muted,
      align: "center",
      margin: 0,
      isTextBox: true,
    });
    if (i < 3) {
      s.addText("→", {
        x: x + 2.05,
        y: 2.15,
        w: 0.3,
        h: 0.4,
        fontSize: 20,
        color: COL.muted,
        margin: 0,
        isTextBox: true,
      });
    }
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5,
    y: 4.05,
    w: 9.0,
    h: 0.7,
    fill: { color: COL.light },
  });
  s.addText(
    "A cell is one model on one task in one language. If clean accuracy is below 0.80 or the logit difference is below 0.5, I do not localise it.",
    {
      x: 0.75,
      y: 4.05,
      w: 8.5,
      h: 0.7,
      fontSize: 14,
      color: COL.ink,
      margin: 0,
      valign: "middle",
      isTextBox: true,
    }
  );

  s.addNotes(notesFor(3));
}

function slide4(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Most small models cannot do the Hindi tasks", { placeholder: "title" });

  const models = [
    "gpt2",
    "pythia-160m",
    "pythia-410m",
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Llama-3.2-1B",
    "bloom-560m",
    "bloom-1b1",
    "bloom-1b7",
  ];
  const cols = [
    { task: "ioi", lang: "en", label: "IOI\nen" },
    { task: "capitals", lang: "en", label: "Cap\nen" },
    { task: "agreement", lang: "en", label: "Agr\nen" },
    { task: "induction", lang: "en", label: "Ind\nen" },
    { task: "ioi", lang: "hi", label: "IOI\nhi" },
    { task: "capitals", lang: "hi", label: "Cap\nhi" },
    { task: "agreement", lang: "hi", label: "Agr\nhi" },
    { task: "induction", lang: "hi", label: "Ind\nhi" },
    { task: "ioi", lang: "hi_roman", label: "IOI\nrom" },
  ];
  const byKey = {};
  for (const r of DATA.gate) byKey[`${r.model}|${r.task}|${r.lang}`] = r.tier;

  const x0 = 1.7;
  const y0 = 1.15;
  const cw = 0.78;
  const rh = 0.34;

  cols.forEach((c, j) => {
    s.addText(c.label, {
      x: x0 + j * cw,
      y: y0 - 0.05,
      w: cw,
      h: 0.45,
      fontSize: 9,
      color: COL.muted,
      align: "center",
      margin: 0,
      isTextBox: true,
    });
  });

  models.forEach((m, i) => {
    const y = y0 + 0.45 + i * rh;
    s.addText(m, {
      x: 0.35,
      y,
      w: 1.3,
      h: rh - 0.04,
      fontSize: 10,
      color: COL.ink,
      valign: "middle",
      margin: 0,
      isTextBox: true,
    });
    cols.forEach((c, j) => {
      const tier = byKey[`${m}|${c.task}|${c.lang}`];
      const x = x0 + j * cw + 0.08;
      if (!tier) {
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
          x,
          y: y + 0.04,
          w: cw - 0.16,
          h: rh - 0.1,
          fill: { color: "E8EBEE" },
          rectRadius: 0.04,
        });
        return;
      }
      const fill =
        tier === "strict" ? COL.softPass : tier === "first_token" ? COL.softFirst : COL.softFail;
      const mark = tier === "strict" ? "●" : tier === "first_token" ? "◐" : "○";
      const colr = tier === "strict" ? COL.pass : tier === "first_token" ? COL.saffron : COL.fail;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x,
        y: y + 0.04,
        w: cw - 0.16,
        h: rh - 0.1,
        fill: { color: fill },
        rectRadius: 0.04,
      });
      s.addText(mark, {
        x,
        y: y + 0.02,
        w: cw - 0.16,
        h: rh - 0.08,
        fontSize: 11,
        color: colr,
        align: "center",
        valign: "middle",
        margin: 0,
        isTextBox: true,
      });
    });
  });

  s.addText(
    [
      { text: "● passed  ", options: { color: COL.pass } },
      { text: "◐ first token only  ", options: { color: COL.saffron } },
      { text: "○ failed  ", options: { color: COL.fail } },
      { text: "grey = English-only models, not gated on Hindi", options: { color: COL.muted } },
    ],
    { x: 0.5, y: 4.75, w: 9.0, h: 0.28, fontSize: 11, margin: 0, isTextBox: true }
  );

  s.addNotes(notesFor(4));
}

function slide5(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("On attention heads, the two methods give the same ranking", {
    placeholder: "title",
  });

  const bars = DATA.h1.bars;
  const shortModel = (m) => {
    if (m.startsWith("Qwen2.5-0.5")) return "Q0.5";
    if (m.startsWith("Qwen2.5-1.5")) return "Q1.5";
    if (m.startsWith("Llama")) return "Lla";
    if (m.startsWith("bloom-560")) return "B560";
    if (m.startsWith("bloom-1b1")) return "B1b1";
    if (m.startsWith("bloom-1b7")) return "B1b7";
    if (m === "gpt2") return "gpt2";
    return m.slice(0, 5);
  };
  const shortTask = { agreement: "agr", capitals: "cap", ioi: "ioi", induction: "ind" };
  const labels = bars.map((b) => `${shortModel(b.model)} ${shortTask[b.task] || b.task}/${b.lang}`);
  const values = bars.map((b) => Math.round(b.rho * 1000) / 1000);
  const H1_THRESHOLD = DATA.h1.threshold;
  const barColors = bars.map((b) => (b.rho < H1_THRESHOLD ? COL.fail : COL.teal));
  const nMiss = barColors.filter((c) => c === COL.fail).length;

  s.addChart(
    pres.charts.BAR,
    [{ name: "Spearman ρ (AP, AttrP)", labels, values }],
    {
      x: 0.4,
      y: 1.0,
      w: 7.0,
      h: 3.65,
      barGrouping: "clustered",
      chartColors: barColors,
      showValue: false,
      showLegend: false,
      catAxisLabelColor: COL.muted,
      catAxisLabelFontSize: 8,
      catAxisLabelFontFace: "+mn-lt",
      valAxisMinVal: 0.8,
      valAxisMaxVal: 1.0,
      valAxisMajorUnit: 0.05,
      valAxisLabelFormatCode: "0.00",
      valAxisLabelColor: COL.muted,
      valAxisLabelFontSize: 10,
      valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "D0D5D8", size: 0.5 },
      catGridLine: { style: "none" },
    }
  );

  s.addText(
    [
      { text: "■ ", options: { color: COL.teal } },
      { text: `at or above ${fmt(H1_THRESHOLD, 2)} (${bars.length - nMiss})     `, options: { color: COL.ink } },
      { text: "■ ", options: { color: COL.fail } },
      { text: `below the preregistered ${fmt(H1_THRESHOLD, 2)} (${nMiss})     `, options: { color: COL.ink } },
      { text: "axis starts at 0.80", options: { color: COL.muted } },
    ],
    { x: 0.6, y: 4.72, w: 6.8, h: 0.28, fontSize: 11, margin: 0, valign: "middle", isTextBox: true }
  );

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.5,
    y: 1.2,
    w: 2.1,
    h: 3.4,
    fill: { color: COL.card },
    rectRadius: 0.1,
  });
  s.addText(`${H1_PASS} / ${H1_N}`, {
    x: 7.6,
    y: 1.5,
    w: 1.9,
    h: 0.7,
    fontSize: 32,
    bold: true,
    color: COL.teal,
    align: "center",
    margin: 0,
    isTextBox: true,
  });
  s.addText(`cells at or\nabove ${H1_THR}`, {
    x: 7.6,
    y: 2.25,
    w: 1.9,
    h: 0.6,
    fontSize: 13,
    color: COL.ink,
    align: "center",
    margin: 0,
    isTextBox: true,
  });
  s.addText(`Misses:\nQwen-0.5B agr/en\n${MISS_A}\nQwen-1.5B agr/en\n${MISS_B}`, {
    x: 7.65,
    y: 3.05,
    w: 1.8,
    h: 1.3,
    fontSize: 12,
    color: COL.fail,
    align: "center",
    margin: 0,
    isTextBox: true,
  });

  s.addNotes(notesFor(5));
}

function slide6(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Where Hindi can be tested, it agrees as well as English", {
    placeholder: "title",
  });

  const pairs = DATA.h3.pairs;
  s.addChart(
    pres.charts.BAR,
    [
      {
        name: "English",
        labels: pairs.map((p) => p.model),
        values: pairs.map((p) => Math.round(p.rho_en * 1000) / 1000),
      },
      {
        name: "Hindi",
        labels: pairs.map((p) => p.model),
        values: pairs.map((p) => Math.round(p.rho_hi * 1000) / 1000),
      },
    ],
    {
      x: 0.4,
      y: 1.05,
      w: 5.8,
      h: 3.5,
      barGrouping: "clustered",
      chartColors: [COL.teal, COL.saffron],
      showLegend: true,
      legendPos: "b",
      legendFontSize: 12,
      legendColor: COL.ink,
      catAxisLabelColor: COL.ink,
      catAxisLabelFontSize: 12,
      catAxisLabelFontFace: "+mn-lt",
      valAxisMinVal: 0.8,
      valAxisMaxVal: 1.0,
      valAxisMajorUnit: 0.05,
      valAxisLabelFormatCode: "0.00",
      valAxisLabelColor: COL.muted,
      valAxisLabelFontSize: 11,
      valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "D0D5D8", size: 0.5 },
      catGridLine: { style: "none" },
      showValue: true,
      dataLabelFormatCode: "0.00",
      dataLabelPosition: "outEnd",
      dataLabelFontSize: 11,
      dataLabelFontFace: "+mn-lt",
      dataLabelColor: COL.ink,
    }
  );

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.5,
    y: 1.2,
    w: 3.1,
    h: 3.2,
    fill: { color: COL.card },
    rectRadius: 0.1,
  });
  s.addText("Equivalence test", {
    x: 6.7,
    y: 1.4,
    w: 2.7,
    h: 0.35,
    fontSize: 13,
    bold: true,
    color: COL.muted,
    margin: 0,
    isTextBox: true,
  });
  s.addText("Equivalent\nwithin ±0.05", {
    x: 6.7,
    y: 1.85,
    w: 2.7,
    h: 0.85,
    fontSize: 20,
    bold: true,
    color: COL.teal,
    margin: 0,
    isTextBox: true,
  });
  s.addText(
    `Pooled Hindi − English\n+${fmt(DATA.h3.pooled_mean_diff, 3)}\n95% CI [${fmt(DATA.h3.ci95[0], 3)}, ${fmt(DATA.h3.ci95[1], 3)}]\n\nCaveat: 3 pairs,\none family (BLOOM),\nagreement only.`,
    {
      x: 6.7,
      y: 2.85,
      w: 2.7,
      h: 1.4,
      fontSize: 12,
      color: COL.ink,
      margin: 0,
      isTextBox: true,
    }
  );

  s.addNotes(notesFor(6));
}

function slide7(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("On MLPs, the disagreement comes from layer 0", { placeholder: "title" });

  const mlp = DATA.mlp_layers.gpt2_ioi;
  const labels = mlp.ap.map((_, i) => String(i));
  s.addChart(
    pres.charts.LINE,
    [
      { name: "Activation patching", labels, values: mlp.ap.map((v) => Math.round(v * 1000) / 1000) },
      { name: "Attribution patching", labels, values: mlp.attr.map((v) => Math.round(v * 1000) / 1000) },
    ],
    {
      x: 0.3,
      y: 1.0,
      w: 6.5,
      h: 3.7,
      chartColors: [COL.teal, COL.saffron],
      showLegend: true,
      legendPos: "b",
      legendFontSize: 12,
      lineDataSymbol: "circle",
      lineDataSymbolSize: 8,
      showCatAxisTitle: true,
      catAxisTitle: "MLP layer",
      catAxisTitleColor: COL.muted,
      catAxisTitleFontSize: 11,
      catAxisTitleFontFace: "+mn-lt",
      catAxisLabelColor: COL.ink,
      catAxisLabelFontSize: 11,
      catAxisLabelFontFace: "+mn-lt",
      showValAxisTitle: true,
      valAxisTitle: "Effect on logit difference",
      valAxisTitleColor: COL.muted,
      valAxisTitleFontSize: 11,
      valAxisTitleFontFace: "+mn-lt",
      valAxisTitleRotate: -90,
      valAxisLabelColor: COL.muted,
      valAxisLabelFontSize: 11,
      valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "D0D5D8", size: 0.5 },
      catGridLine: { style: "none" },
    }
  );

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.95,
    y: 1.15,
    w: 2.65,
    h: 3.4,
    fill: { color: COL.card },
    rectRadius: 0.1,
  });
  s.addText("GPT-2 IOI\n(mean of 5 seeds)", {
    x: 7.1,
    y: 1.35,
    w: 2.35,
    h: 0.55,
    fontSize: 13,
    bold: true,
    color: COL.ink,
    margin: 0,
    isTextBox: true,
  });
  s.addText(
    `Layer 0\nAP  ${fmt(mlp.ap[0], 2)}\nAttrP  ${fmt(mlp.attr[0], 2)}\n\nLayers 1–11\nmatch within 0.01\n\nSame pattern:\nQwen-0.5B, Pythia\nIOI (post hoc).`,
    {
      x: 7.1,
      y: 2.05,
      w: 2.35,
      h: 2.3,
      fontSize: 13,
      color: COL.ink,
      margin: 0,
      isTextBox: true,
    }
  );

  s.addNotes(notesFor(7));
}

function slide8(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Integrated gradients fixes layer 0 but changes head ranking", {
    placeholder: "title",
  });

  const g = DATA.eap_examples.gpt2_ioi;
  const SWEEP_RHO_SHORT = ["5", "10", "20"].map((k) => fmt(SWEEP[k].mean_head_rho, 2)).join(" / ");

  const cards = [
    {
      x: 0.5,
      title: "MLP agreement (GPT-2 IOI)",
      before: fmt(g.mlp_r, 2),
      after: fmt(g.eap_mlp_r, 2),
      verdict: "Fixes the layer-0 failure",
      color: COL.pass,
    },
    {
      x: 5.2,
      title: "Head ranking (GPT-2 IOI)",
      before: fmt(g.rho, 2),
      after: fmt(g.eap_rho, 2),
      verdict: "Head ranking no longer matches AP",
      color: COL.fail,
    },
  ];
  cards.forEach((c) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: c.x,
      y: 1.15,
      w: 4.3,
      h: 2.4,
      fill: { color: COL.card },
      rectRadius: 0.1,
    });
    s.addText(c.title, {
      x: c.x + 0.25,
      y: 1.3,
      w: 3.8,
      h: 0.35,
      fontSize: 14,
      bold: true,
      color: COL.ink,
      margin: 0,
      isTextBox: true,
    });
    s.addText(
      [
        { text: "AttrP  ", options: { color: COL.saffron, bold: true } },
        { text: c.before, options: { color: COL.ink, fontSize: 28, bold: true } },
        { text: "   →   ", options: { color: COL.muted, fontSize: 18 } },
        { text: "EAP-IG  ", options: { color: COL.plum, bold: true, fontSize: 14 } },
        { text: c.after, options: { color: COL.ink, fontSize: 28, bold: true } },
      ],
      { x: c.x + 0.25, y: 1.85, w: 3.8, h: 0.7, margin: 0, isTextBox: true }
    );
    s.addText(c.verdict, {
      x: c.x + 0.25,
      y: 2.75,
      w: 3.8,
      h: 0.5,
      fontSize: 14,
      bold: true,
      color: c.color,
      margin: 0,
      isTextBox: true,
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5,
    y: 3.75,
    w: 9.0,
    h: 1.15,
    fill: { color: COL.charcoal },
    rectRadius: 0.08,
  });
  s.addText(
    [
      {
        text: `EAP-IG at 5 steps costs about ${EAP_X}× AttrP. `,
        options: { bold: true, color: COL.saffron },
      },
      {
        text: `Across the grid, mean head correlation drops by ${fmt(-DATA.h5.head_rho_gain_mean, 2)}, and the MLP gain is indistinguishable from zero.`,
        options: { color: COL.white },
      },
    ],
    { x: 0.7, y: 3.88, w: 8.6, h: 0.45, fontSize: 13, margin: 0, isTextBox: true }
  );
  s.addText(
    `More steps do not help: mean head correlation is ${SWEEP_RHO_SHORT} at 5 / 10 / 20 steps (seed 0).`,
    {
      x: 0.7,
      y: 4.4,
      w: 8.6,
      h: 0.35,
      fontSize: 13,
      color: "CAD5DC",
      margin: 0,
      isTextBox: true,
    }
  );

  s.addNotes(notesFor(8));
}

function slide9(pres, C) {
  const s = pres.addSlide({ masterName: "CLOSE_DARK", sectionTitle: "Close" });
  s.addText("What I would do in practice", { placeholder: "title" });

  const rules = [
    {
      n: "1",
      t: "Use attribution patching for attention heads",
      d: `It agreed in ${H1_PASS} of ${H1_N} cells, and it is about ${AP_X} times cheaper.`,
    },
    {
      n: "2",
      t: "Check early MLPs with real activation patching",
      d: "Layer 0 can flip sign under attribution patching on IOI-type circuits.",
    },
    {
      n: "3",
      t: "Gate the task before you localise",
      d: "Hindi IOI fails in all 6 models gated on Hindi, capitals in 5 of 6.",
    },
  ];
  rules.forEach((r, i) => {
    const y = 1.2 + i * 0.95;
    s.addShape(pres.shapes.OVAL, {
      x: 0.6,
      y: y,
      w: 0.5,
      h: 0.5,
      fill: { color: COL.saffron },
    });
    s.addText(r.n, {
      x: 0.6,
      y: y + 0.08,
      w: 0.5,
      h: 0.35,
      fontSize: 18,
      bold: true,
      color: COL.white,
      align: "center",
      margin: 0,
      isTextBox: true,
    });
    s.addText(r.t, {
      x: 1.3,
      y: y,
      w: 8.0,
      h: 0.35,
      fontSize: 18,
      bold: true,
      color: COL.white,
      margin: 0,
      isTextBox: true,
    });
    s.addText(r.d, {
      x: 1.3,
      y: y + 0.35,
      w: 8.0,
      h: 0.35,
      fontSize: 14,
      color: "B0B8C0",
      margin: 0,
      isTextBox: true,
    });
  });

  s.addText(
    "Limits: Hindi evidence is mostly BLOOM agreement. Gemma-3 was dropped for tooling. Seeds resample prompts. I reviewed the templates myself.",
    {
      x: 0.6,
      y: 4.85,
      w: 8.8,
      h: 0.35,
      fontSize: 11,
      color: "8A939B",
      margin: 0,
      isTextBox: true,
    }
  );

  s.addNotes(notesFor(9));
}

/* -------------------- Backup -------------------- */

function backup1(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: gate coverage by task and language", { placeholder: "title" });

  const rows = [
    [
      { text: "Condition", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
      { text: "Passed", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
      { text: "First tok", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
      { text: "Failed", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
    ],
  ];
  const order = [
    ["ioi", "en"],
    ["capitals", "en"],
    ["agreement", "en"],
    ["induction", "en"],
    ["ioi", "hi"],
    ["capitals", "hi"],
    ["agreement", "hi"],
    ["induction", "hi"],
    ["ioi", "hi_roman"],
  ];
  const counts = {};
  for (const r of DATA.gate) {
    const k = `${r.task}|${r.lang}`;
    if (!counts[k]) counts[k] = { strict: 0, first_token: 0, fail: 0 };
    counts[k][r.tier] = (counts[k][r.tier] || 0) + 1;
  }
  for (const [task, lang] of order) {
    const c = counts[`${task}|${lang}`] || { strict: 0, first_token: 0, fail: 0 };
    rows.push([
      { text: `${task} / ${lang}`, options: { color: COL.ink } },
      { text: String(c.strict || 0), options: { color: COL.pass } },
      { text: String(c.first_token || 0), options: { color: COL.saffron } },
      { text: String(c.fail || 0), options: { color: COL.fail } },
    ]);
  }
  s.addTable(rows, {
    x: 1.2,
    y: 1.1,
    w: 7.6,
    colW: [2.8, 1.6, 1.6, 1.6],
    border: [{ pt: 0.5, color: "D0D5D8" }],
    fontFace: "Calibri",
    fontSize: 13,
    align: "center",
    valign: "middle",
    rowH: 0.36,
  });

  s.addNotes(notesFor(10));
}

function backup2(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: noising still agrees on heads", { placeholder: "title" });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 1.5,
    y: 1.6,
    w: 7.0,
    h: 2.4,
    fill: { color: COL.card },
    rectRadius: 0.1,
  });
  s.addText(fmt(DATA.noising_mean_rho, 2), {
    x: 1.5,
    y: 1.9,
    w: 7.0,
    h: 0.9,
    fontSize: 54,
    bold: true,
    color: COL.teal,
    align: "center",
    margin: 0,
    isTextBox: true,
  });
  s.addText("Mean head Spearman correlation (AP, AttrP) under noising", {
    x: 1.7,
    y: 2.9,
    w: 6.6,
    h: 0.5,
    fontSize: 15,
    color: COL.ink,
    align: "center",
    margin: 0,
    isTextBox: true,
  });
  s.addText(
    "Slightly lower than denoising, same qualitative story. Qwen and Llama agreement noising dips to about 0.76–0.86.",
    {
      x: 1.0,
      y: 4.3,
      w: 8.0,
      h: 0.5,
      fontSize: 13,
      color: COL.muted,
      align: "center",
      margin: 0,
      isTextBox: true,
    }
  );
  s.addNotes(notesFor(11));
}

function backup3(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: design changes and fertility", { placeholder: "title" });

  s.addText("Amendment 2 (gate phase, before any grid result)", {
    x: 0.5,
    y: 1.1,
    w: 4.5,
    h: 0.35,
    fontSize: 14,
    bold: true,
    color: COL.ink,
    margin: 0,
    isTextBox: true,
  });
  s.addText(
    [
      { text: "Gemma-3 replaced by BLOOM (TL 2.15)", options: { breakLine: true } },
      { text: "Added BLOOM-1b1 and 1b7 after first gate", options: { breakLine: true } },
      { text: "Corrupted-LD rule: max(0.25, 0.5×clean)", options: { breakLine: true } },
      { text: "First-token tier for multi-token answers", options: { breakLine: true } },
      { text: "Noising sign fix (smoke-test catch)", options: {} },
    ],
    {
      x: 0.5,
      y: 1.55,
      w: 4.5,
      h: 2.2,
      fontSize: 13,
      color: COL.ink,
      paraSpaceAfter: 6,
      margin: 0,
      valign: "top",
      isTextBox: true,
    }
  );

  s.addText("Fertility and clean LD (observational)", {
    x: 5.3,
    y: 1.1,
    w: 4.2,
    h: 0.35,
    fontSize: 14,
    bold: true,
    color: COL.ink,
    margin: 0,
    isTextBox: true,
  });
  const h4 = DATA.h4 || {};
  const tpw = h4.rho_vs_tokens_per_word || {};
  const cld = h4.rho_vs_clean_ld || {};
  s.addText(
    [
      {
        text: `Spearman with tokens/word = ${fmt(tpw.rho, 2)}`,
        options: { breakLine: true },
      },
      {
        text: `CI [${fmt(tpw.ci_lo, 2)}, ${fmt(tpw.ci_hi, 2)}]`,
        options: { breakLine: true, color: COL.muted },
      },
      { text: "", options: { breakLine: true } },
      {
        text: `Spearman with clean LD = ${fmt(cld.rho, 2)}`,
        options: { breakLine: true },
      },
      {
        text: `CI [${fmt(cld.ci_lo, 2)}, ${fmt(cld.ci_hi, 2)}]`,
        options: { breakLine: true, color: COL.muted },
      },
      { text: "", options: { breakLine: true } },
      {
        text: "Observational only. Fertility is entangled with model and task; no high-fertility Hindi cells gated.",
        options: { color: COL.muted },
      },
    ],
    {
      x: 5.3,
      y: 1.55,
      w: 4.2,
      h: 3.0,
      fontSize: 13,
      color: COL.ink,
      paraSpaceAfter: 4,
      margin: 0,
      valign: "top",
      isTextBox: true,
    }
  );

  s.addNotes(notesFor(12));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
