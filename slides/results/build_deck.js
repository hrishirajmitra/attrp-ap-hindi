#!/usr/bin/env node
/**
 * Build slides/results/results_deck.pptx from deck_data.json.
 * Structured deck: theme + layouts + native charts + speaker notes.
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
    accent1: "0D7377", // teal — activation patching
    accent2: "E07A3D", // saffron — attribution patching
    accent3: "6B3A6B", // plum — EAP-IG
    accent4: "C0392B", // fail red
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

function notes(lines) {
  return lines.filter(Boolean).join("\n\n");
}

async function main() {
  const pres = new pptxgen();
  pres.defineLayout({ name: "LAYOUT_16x9", width: 10, height: 5.625 });
  pres.layout = "LAYOUT_16x9";
  pres.author = DATA.meta.author;
  pres.title = "When can the cheap patching method replace the expensive one?";
  pres.subject = "AttrP vs AP · English/Hindi multi-cell study";
  pres.company = "IIIT Hyderabad";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

  const C = pres.SchemeColor;

  // Layouts
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
            fontSize: 26,
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

  // Sections
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

  // Backup for Q&A
  pres.addSection({ title: "Backup" });
  backup1(pres, C);
  backup2(pres, C);
  backup3(pres, C);

  await pres.writeFile({ fileName: OUT });

  // Apply theme colors into the package (needs NODE_PATH to this folder's node_modules)
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

  // Export speaker notes markdown
  writeSpeakerNotesMd();
  console.log("wrote", OUT);
}

/* -------------------- Slides -------------------- */

function slide1(pres, C) {
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addText("When can the cheap patching method\nreplace the expensive one?", {
    placeholder: "title",
  });
  s.addText("Attribution patching vs activation patching · English and Hindi · 9 models", {
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
  // Accent dots for method colours
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
  s.addNotes(
    notes([
      "Open with the question on the slide — don't read the subtitle. Pause one beat.",
      "One sentence of setup: activation patching is the causal gold standard for finding which heads and MLPs matter, but it is expensive. Attribution patching is the cheap first-order shortcut everyone uses when models get bigger.",
      "Today's answer, upfront: for attention heads, yes — almost always. For MLP outputs, not always, and the failure is usually one early layer. Hindi only where the model can do the task.",
      "Transition: so what is the trade-off, exactly?",
      "Q: Is this a new method? A: No — we measure when an existing cheap method matches the expensive one, across languages.",
    ])
  );
}

function slide2(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  s.addText(`Activation patching is exact — and about ${fmt(DATA.cost.ap_over_attrp, 0)}× slower`, {
    placeholder: "title",
  });

  // Two method cards
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
      { text: "Causal gold standard", options: { bold: true, breakLine: true } },
      { text: "One forward pass per head / MLP", options: { breakLine: true } },
      { text: "Scales with #components × prompts", options: { breakLine: true } },
      { text: `~${fmt(DATA.cost.mean_t_ap, 0)}s per cell on a 2080 Ti`, options: {} },
    ],
    { x: 0.75, y: 1.85, w: 3.8, h: 1.7, fontSize: 14, color: COL.ink, paraSpaceAfter: 6, margin: 0, isTextBox: true }
  );
  s.addText(
    [
      { text: "First-order gradient shortcut", options: { bold: true, breakLine: true } },
      { text: "One forward + one backward pass", options: { breakLine: true } },
      { text: "Ranks all components at once", options: { breakLine: true } },
      { text: `~${fmt(DATA.cost.mean_t_attr, 1)}s per cell — ~${fmt(DATA.cost.ap_over_attrp, 0)}× faster`, options: {} },
    ],
    { x: 5.45, y: 1.85, w: 3.8, h: 1.7, fontSize: 14, color: COL.ink, paraSpaceAfter: 6, margin: 0, isTextBox: true }
  );

  // Big cost callout
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
      { text: `~${fmt(DATA.cost.ap_over_attrp, 0)}×  `, options: { bold: true, fontSize: 28, color: COL.saffron } },
      {
        text: "wall-time of AP vs AttrP on our grid. When do they agree?",
        options: { fontSize: 15, color: COL.white },
      },
    ],
    { x: 0.75, y: 4.25, w: 8.5, h: 0.55, margin: 0, valign: "middle", isTextBox: true }
  );

  s.addNotes(
    notes([
      "Point to the left card first: activation patching literally copies a component's activation from a clean run into a corrupted run. Exact, slow.",
      "Right card: attribution patching is a Taylor approximation — one backward pass, all components at once.",
      `The ~${fmt(DATA.cost.ap_over_attrp, 0)}× is measured wall time on our Ada 2080 Ti grid — the median per-run AP/AttrP time ratio over the ${DATA.cost.n_runs_timed} core denoising runs (the mean times on the cards, ~${fmt(DATA.cost.mean_t_ap, 0)} s vs ~${fmt(DATA.cost.mean_t_attr, 1)} s, give the same ~${fmt(DATA.cost.ap_over_attrp_ratio_of_means, 0)}×). Not a theoretical FLOP count. Say that if asked.`,
      "End on the question in the banner. Transition: here is what we actually ran.",
      "Q: Why not just always use activation patching? A: At 1.5B+ parameters and many cells, it is the bottleneck. The cheap method is what people ship.",
    ])
  );
}

function slide3(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  s.addText("What we ran: gate first, then patch", { placeholder: "title" });

  const steps = [
    { n: "1", t: "9 models", d: "GPT-2, Pythia, Qwen, Llama, BLOOM\n(≤2B / 11 GB)" },
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
    "Rule: if the model cannot do the task (clean accuracy < 0.80, clean logit-diff < 0.5), we do not localise it.",
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

  s.addNotes(
    notes([
      "Walk the four boxes left to right. Emphasise the gate: we refuse to localise behaviour the model does not have.",
      "Gemma-3 was in the proposal; TransformerLens 2.15 cannot load it on our Ada torch build, so BLOOM replaced it — logged in Amendment 2, before any grid result.",
      `${DATA.meta.n_runs} runs, zero failures. About four hours of wall time across irel and research accounts.`,
      "Transition: the gate itself is the first finding — especially for Hindi.",
      "Q: Why five seeds? A: They resample prompts, not weights. CIs reflect prompt sampling.",
    ])
  );
}

function slide4(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Finding 1: the gate is the Hindi result", { placeholder: "title" });

  // Build a compact gate matrix: models × key conditions
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

  // Legend + takeaway
  s.addText(
    [
      { text: "● strict  ", options: { color: COL.pass } },
      { text: "◐ first-token  ", options: { color: COL.saffron } },
      { text: "○ fail  ", options: { color: COL.fail } },
      { text: "grey = not gated (English-only models)", options: { color: COL.muted } },
    ],
    { x: 0.5, y: 4.75, w: 9.0, h: 0.28, fontSize: 11, margin: 0, isTextBox: true }
  );

  s.addNotes(
    notes([
      `This grid is the Hindi headline. Point to the right half: Hindi IOI and romanised IOI are red in all six models gated on Hindi; Hindi capitals is red except one first-token pass (Llama, the orange half-circle). In the ${DATA.hindi_fail_acc.n_cells} red cells of those three columns clean accuracy is ${fmt(DATA.hindi_fail_acc.min, 2)}–${fmt(DATA.hindi_fail_acc.max, 2)} — the models cannot do the task.`,
      "Green Hindi cells: agreement on BLOOM (three sizes) and induction on BLOOM/Llama. That is the entire Hindi evidence for AttrP≈AP.",
      "Grey cells are English-only models we never gated on Hindi.",
      "Native-speaker review of Hindi templates: one place-word fix; re-gate unchanged. So this is mostly model competence, not broken templates.",
      "Transition: where the gate passes, do the methods agree on heads?",
      "Q: Why so little Hindi? A: That is the finding. You cannot localise a behaviour the model does not have.",
    ])
  );
}

function slide5(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Finding 2: for heads, the cheap method works", { placeholder: "title" });

  const bars = DATA.h1.bars;
  // Sort for readable chart: by model then task
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
  const H1_THRESHOLD = DATA.h1.threshold; // preregistered 0.90
  // One colour per bar: pptxgenjs colours a single-series bar chart per point when chartColors
  // has more than one entry. Fail red for seed-mean rho below the threshold, teal otherwise.
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

  // Legend-style caption for the bar colours (the 0.90 gridline is the threshold)
  s.addText(
    [
      { text: "■ ", options: { color: COL.teal } },
      { text: `ρ ≥ ${fmt(H1_THRESHOLD, 2)} (${bars.length - nMiss} cells)     `, options: { color: COL.ink } },
      { text: "■ ", options: { color: COL.fail } },
      { text: `red = below the preregistered ${fmt(H1_THRESHOLD, 2)} (${nMiss} cells)     `, options: { color: COL.ink } },
      { text: "axis starts at 0.80", options: { color: COL.muted } },
    ],
    { x: 0.6, y: 4.72, w: 6.8, h: 0.28, fontSize: 11, margin: 0, valign: "middle", isTextBox: true }
  );

  // Threshold callout
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.5,
    y: 1.2,
    w: 2.1,
    h: 3.4,
    fill: { color: COL.card },
    rectRadius: 0.1,
  });
  s.addText("22 / 24", {
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
  s.addText("core cells\nρ ≥ 0.90", {
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
  s.addText("Misses:\nQwen-0.5B agr/en\n0.86\nQwen-1.5B agr/en\n0.88", {
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

  s.addNotes(
    notes([
      "This is H1. Head ranking agreement between AP and AttrP. Threshold 0.90 was preregistered.",
      "22 of 24 strict-tier core cells pass. The two misses are Qwen English agreement — still high, just under the bar.",
      "Noising direction is a bit lower (mean ρ ≈ 0.93) but the same story.",
      "Transition: among the cells that pass the Hindi gate, is AttrP worse in Hindi?",
      "Q: Is 0.90 magic? A: No, but it was fixed in the preregistration. The two misses are 0.86 and 0.88; the other 22 cells are 0.91–0.995. Do not lean on the restricted ρ (heads with |AP|>0.05): only a handful of heads pass that cut-off (median 4 per run), and it is higher than the plain ρ in just 11 of the 22 cells where it is defined.",
    ])
  );
}

function slide6(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Finding 3: where testable, Hindi matches English", {
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
  s.addText("H3 verdict", {
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
    `Pooled Hindi−English\nρ = +${fmt(DATA.h3.pooled_mean_diff, 3)}\n95% CI [${fmt(DATA.h3.ci95[0], 3)}, ${fmt(DATA.h3.ci95[1], 3)}]\n\nCaveat: 3 pairs,\none family (BLOOM),\nagreement only.`,
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

  s.addNotes(
    notes([
      "Only core task where we can pair English and Hindi on the same model with a strict gate: BLOOM agreement, three sizes. (Induction also pairs, on BLOOM and Llama, but it is appendix-only by the preregistration; Hindi is within 0.02 of English there too.)",
      `Bars are almost the same; the axis starts at 0.80. Equivalence test (TOST, preregistered margin ±0.05): pooled Hindi−English difference +${fmt(DATA.h3.pooled_mean_diff, 3)}, 90% CI [${fmt(DATA.summary.H3.ci90[0], 3)}, +${fmt(DATA.summary.H3.ci90[1], 3)}], 95% CI [${fmt(DATA.h3.ci95[0], 3)}, +${fmt(DATA.h3.ci95[1], 3)}] — both inside the margin, so the verdict is equivalent.`,
      "Say the caveat out loud: three pairs, one family. We do not claim this for Hindi IOI — those cells never gated.",
      "Transition: heads look good. What about MLPs?",
      "Q: Is BLOOM special? A: It is the only family with usable Hindi agreement. That is a limit, not a virtue.",
    ])
  );
}

function slide7(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Finding 4: the MLP failure is one early layer", { placeholder: "title" });

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

  s.addNotes(
    notes([
      "H2 said MLP AttrP would anti-correlate in most cells. Rejected: only 2 of 24 core cells. The famous GPT-2 IOI failure is real — and it is almost entirely MLP 0.",
      "Walk the chart: teal and saffron overlap from layer 1 onward; only layer 0 flips sign.",
      "Same layer-0 pattern in Qwen-0.5B and both Pythias on IOI. Labelled post hoc — we looked after seeing H2 fail.",
      `BLOOM IOI MLPs actually agree: Pearson r(AP, AttrP) ≈ +${fmt(Math.min(...Object.values(DATA.bloom_ioi_mlp_r)), 3)} or higher in all three sizes. So this is circuit-specific, not universal.`,
      "Transition: can integrated gradients fix layer 0 without breaking heads?",
      "Q: Why MLP 0? A: Early nonlinearities; the linear AttrP approximation is worst there. We did not claim a mechanism beyond that.",
    ])
  );
}

function slide8(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
  s.addText("Finding 5: EAP-IG swaps one error for another", {
    placeholder: "title",
  });

  const g = DATA.eap_examples.gpt2_ioi;
  // EAP-IG step sweep (seed 0, core denoising cells): grid means per step count, from summary.json
  const sweep = DATA.h5.step_sweep_seed0;
  const SWEEP_RHO = ["5", "10", "20"].map((k) => fmt(sweep[k].mean_head_rho, 2)).join(" / ");
  const SWEEP_MLP = ["5", "10", "20"].map((k) => fmt(sweep[k].mean_mlp_r, 2)).join(" / ");

  // Two comparison cards
  const cards = [
    {
      x: 0.5,
      title: "MLP agreement (GPT-2 IOI)",
      before: fmt(g.mlp_r, 2),
      after: fmt(g.eap_mlp_r, 2),
      verdict: "Fixes layer-0 failure",
      color: COL.pass,
    },
    {
      x: 5.2,
      title: "Head ranking (GPT-2 IOI)",
      before: fmt(g.rho, 2),
      after: fmt(g.eap_rho, 2),
      verdict: "Breaks head ranking",
      color: COL.fail,
    },
  ];
  cards.forEach((c) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: c.x,
      y: 1.15,
      w: 4.3,
      h: 2.55,
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
      { x: c.x + 0.25, y: 1.9, w: 3.8, h: 0.7, margin: 0, isTextBox: true }
    );
    s.addText(c.verdict, {
      x: c.x + 0.25,
      y: 2.9,
      w: 3.8,
      h: 0.4,
      fontSize: 16,
      bold: true,
      color: c.color,
      margin: 0,
      isTextBox: true,
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5,
    y: 3.95,
    w: 9.0,
    h: 0.95,
    fill: { color: COL.charcoal },
    rectRadius: 0.08,
  });
  s.addText(
    [
      {
        text: `Cost: EAP-IG (5 steps) is ~${fmt(DATA.cost.eap_over_attrp, 1)}× AttrP. `,
        options: { bold: true, color: COL.saffron },
      },
      {
        text: `Across the grid, mean head ρ drops by ${fmt(-DATA.h5.head_rho_gain_mean, 2)}; MLP gain is ≈ 0 (CI crosses zero). Not a free upgrade. More steps do not help: mean head ρ is ${SWEEP_RHO} at 5 / 10 / 20 steps (seed 0).`,
        options: { color: COL.white },
      },
    ],
    { x: 0.7, y: 4.1, w: 8.6, h: 0.7, fontSize: 13, margin: 0, isTextBox: true }
  );

  s.addNotes(
    notes([
      "H5 asked whether EAP-IG repairs MLP agreement. Preregistered verdict: no (mean MLP gain CI crosses zero).",
      "Exploratory look at GPT-2 IOI: EAP-IG does fix MLP 0 (−0.93 → +0.95) but head ρ collapses 0.99 → 0.40. That trade is the point of this slide.",
      "Sanity check on the code: with a single gradient evaluation at α=0 (the corrupted-input end of the path for denoising, the clean end for noising) EAP-IG reduces to AttrP. scripts/test_eap_equiv.py asserts this to 1e-5 on GPT-2 (IOI and agreement, both directions). So the head drop comes from integrating along the path, not from a different code path.",
      `Cost: ~${fmt(DATA.cost.eap_over_attrp, 1)}× is the median per-run EAP-IG/AttrP wall-time ratio over the ${DATA.cost.n_runs_timed} core denoising runs, at 5 steps.`,
      "Practitioner takeaway coming next: use AttrP for heads; pay for AP on early MLPs when it matters.",
      `Q: More IG steps? A: We swept 5, 10 and 20 steps on seed 0 (${sweep["5"].n} core cells). The grid means barely move: head ρ ${SWEEP_RHO}, MLP r ${SWEEP_MLP}. Single cells shift by up to 0.09 in head ρ between step counts, far less than the 0.36 drop, so it is not under-integration. Per-cell MLP r is noisier (shifts up to 0.33).`,
    ])
  );
}

function slide9(pres, C) {
  const s = pres.addSlide({ masterName: "CLOSE_DARK", sectionTitle: "Close" });
  s.addText("Takeaway: a three-line rule", { placeholder: "title" });

  const rules = [
    { n: "1", t: "Use AttrP for attention heads", d: `ρ ≥ 0.90 in 22/24 core cells; ~${fmt(DATA.cost.ap_over_attrp, 0)}× cheaper than AP.` },
    { n: "2", t: "Check early MLPs with real AP", d: "Layer 0 can flip sign under AttrP (IOI-type circuits)." },
    { n: "3", t: "Gate the task before localising", d: "Hindi IOI fails in all 6 models gated on Hindi, capitals in 5 of 6 — nothing to patch." },
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
    "Limits: Hindi evidence = BLOOM agreement (+ induction). Gemma-3 not run (TL 2.15). Seeds resample prompts. Native review done.",
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

  s.addNotes(
    notes([
      "Close on the three rules. Do not reopen every finding.",
      "Limits in one breath: Hindi is narrow; Gemma dropped for tooling; seeds are prompt resample; templates reviewed.",
      "Offer backup slides for gate table, noising, and Amendment 2 / fertility.",
      "Thank you — questions.",
      "Q: What would you do next? A: A Hindi-capable larger model with single-token agreement, or a hand-hooked Gemma once TL catches up.",
    ])
  );
}

/* -------------------- Backup -------------------- */

function backup1(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: gate coverage (counts)", { placeholder: "title" });

  const rows = [
    [
      { text: "Condition", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
      { text: "Strict", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
      { text: "First-tok", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
      { text: "Fail", options: { bold: true, color: COL.white, fill: { color: COL.charcoal } } },
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

  s.addNotes("Backup slide. Use if asked for the full gate breakdown by task and language.");
}

function backup2(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: noising direction still agrees on heads", { placeholder: "title" });

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
  s.addText("Mean head Spearman ρ (AP, AttrP) · noising · core cells", {
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
    "Slightly lower than denoising, same qualitative story. Qwen/Llama agreement noising dips to ~0.76–0.86.",
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
  s.addNotes("Backup. Secondary direction from the preregistration.");
}

function backup3(pres, C) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: deviations and H4 (observational)", { placeholder: "title" });

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
      { text: "Gemma-3 → BLOOM (TL 2.15)", options: { breakLine: true } },
      { text: "Added BLOOM-1b1 / 1b7 after first gate", options: { breakLine: true } },
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

  s.addText("H4: fertility & clean LD", {
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
        text: `Spearman(ρ, tokens/word) = ${fmt(tpw.rho, 2)}`,
        options: { breakLine: true },
      },
      {
        text: `CI [${fmt(tpw.ci_lo, 2)}, ${fmt(tpw.ci_hi, 2)}]`,
        options: { breakLine: true, color: COL.muted },
      },
      { text: "", options: { breakLine: true } },
      {
        text: `Spearman(ρ, clean LD) = ${fmt(cld.rho, 2)}`,
        options: { breakLine: true },
      },
      {
        text: `CI [${fmt(cld.ci_lo, 2)}, ${fmt(cld.ci_hi, 2)}]`,
        options: { breakLine: true, color: COL.muted },
      },
      { text: "", options: { breakLine: true } },
      {
        text: "Observational only — fertility entangled with model/task; no high-fertility Hindi cells gated.",
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

  s.addNotes(
    "Backup for design-change questions and the fertility hypothesis. Amendment 2 was written during the gate phase, before any grid result existed. Do not claim more than that: the gate-rule changes came after the first gate numbers, and the noising sign fix was found by a smoke test that compared AP and AttrP (ρ = −0.92) when no study results existed."
  );
}

function writeSpeakerNotesMd() {
  // Reconstruct from a lightweight mirror — notes are in the pptx; also dump a rehearsal script.
  const md = `# Speaker notes · results deck (≈10 min)

Rehearse from the notes pane in PowerPoint, or from this page. Aim ~1 minute per main slide.

## 1. Title
Open with the question. One sentence of setup: AP is exact and expensive; AttrP is the cheap shortcut. Preview the answer: heads yes; early MLPs sometimes no; Hindi only where the model can do the task.

## 2. Trade-off
Left = causal gold standard. Right = first-order Taylor. ~${fmt(DATA.cost.ap_over_attrp, 0)}× is measured wall time on our grid (median per-run ratio; mean ~${fmt(DATA.cost.mean_t_ap, 0)} s vs ~${fmt(DATA.cost.mean_t_attr, 1)} s per run). End on: when do they agree?

## 3. What we ran
Four boxes left to right. Emphasise the competence gate: ${DATA.meta.n_gate_cells} cells gated, ${DATA.meta.n_runnable} runnable (${DATA.meta.n_strict} strict + ${DATA.meta.n_first_token} first-token). Gemma→BLOOM if asked. ${DATA.meta.n_runs} runs, 0 failures.

## 4. Gate / Hindi
Point at the red Hindi block: Hindi IOI and romanised IOI fail in all 6 models gated on Hindi; capitals fails in 5 of 6 (one first-token pass, Llama). Failing-cell accuracy ${fmt(DATA.hindi_fail_acc.min, 2)}–${fmt(DATA.hindi_fail_acc.max, 2)}. Green Hindi = BLOOM agreement (+ induction on BLOOM/Llama). Native review: one place-word fix; re-gate unchanged.

## 5. Heads (H1)
22/24 core cells ρ≥0.90 (teal bars). Two Qwen English-agreement misses, 0.86 and 0.88 (red bars). Threshold was preregistered. Noising mean ρ ${fmt(DATA.noising_mean_rho, 2)}.

## 6. Hindi = English (H3)
Three BLOOM agreement pairs. Pooled Hindi−English +${fmt(DATA.h3.pooled_mean_diff, 3)}, 95% CI [${fmt(DATA.h3.ci95[0], 3)}, +${fmt(DATA.h3.ci95[1], 3)}]: equivalent within ±0.05. Say the caveat out loud.

## 7. MLP layer 0 (H2 exploratory)
Chart: only layer 0 disagrees (AP +${fmt(DATA.mlp_layers.gpt2_ioi.ap[0], 2)} vs AttrP ${fmt(DATA.mlp_layers.gpt2_ioi.attr[0], 2).replace("-", "−")}; layers 1–11 within 0.01). MLP r < 0 in only 2/24 core cells (H2 rejected). Same pattern Qwen-0.5B / Pythia IOI. BLOOM IOI MLPs agree (Pearson r ≈ +${fmt(Math.min(...Object.values(DATA.bloom_ioi_mlp_r)), 3)}) — circuit-specific.

## 8. EAP-IG (H5)
GPT-2 IOI: fixes MLP 0 (r ${fmt(DATA.eap_examples.gpt2_ioi.mlp_r, 2).replace("-", "−")} → +${fmt(DATA.eap_examples.gpt2_ioi.eap_mlp_r, 2)}), breaks head ranking (ρ ${fmt(DATA.eap_examples.gpt2_ioi.rho, 2)} → ${fmt(DATA.eap_examples.gpt2_ioi.eap_rho, 2)}). Grid: mean head ρ drops ${fmt(-DATA.h5.head_rho_gain_mean, 2)}; cost ~${fmt(DATA.cost.eap_over_attrp, 1)}× AttrP (5 steps). Not under-integration: mean head ρ ${["5", "10", "20"].map((k) => fmt(DATA.h5.step_sweep_seed0[k].mean_head_rho, 2)).join(" / ")} at 5 / 10 / 20 steps (seed 0).

## 9. Takeaway
Three rules. One breath of limits. Questions.

## Backup
Gate table · noising mean ρ ${fmt(DATA.noising_mean_rho, 2)} · Amendment 2 (gate phase, before any grid result) + H4 observational (Spearman ${fmt(DATA.h4.rho_vs_tokens_per_word.rho, 2).replace("-", "−")} with tokens/word, +${fmt(DATA.h4.rho_vs_clean_ld.rho, 2)} with clean LD).
`;
  fs.writeFileSync(path.join(ROOT, "speaker_notes.md"), md);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
