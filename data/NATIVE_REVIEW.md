# Native-speaker review checklist (Hindi)

Review before citing Hindi IOI / hand agreement / ne-perfective results.
MultiBLiMP pairs are source-published; spot-check a sample for sense.

## Hindi IOI templates (`src/data_ioi.py`)

Full intended sentence: `{X} और {Y} {PLACE} गए। {Z} ने {OBJ} {IO} को दिया।`  
Prompt ends before `{IO}`.

- [ ] Sounds natural to a native speaker (register OK for a templated probe)
- [ ] Recipient is the only plausible next name
- [ ] `{OBJ}` is an object that can take `को` + `दिया` / equivalent
- [ ] Names in `NAMES_HI` are common given names, not rare/awkward

## Hand agreement / ne-perfective (stretch)

- [ ] Subject-agreeing intransitives: gender on verb matches subject
- [ ] Ne-perfective: verb gender matches **object**, not subject
- [ ] Objects in ne items are clear masculine/feminine nouns

## Sign-off

Reviewer: ______________  Date: ______________
