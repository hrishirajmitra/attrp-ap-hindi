# MultiBLiMP local copies

Source: [jumelet/multiblimp](https://huggingface.co/datasets/jumelet/multiblimp) (MultiBLiMP 1.0).

| File | Language | Pairs |
|------|----------|------:|
| `eng_data.tsv` | English | 770 |
| `hin_data.tsv` | Hindi | 1447 |

Hindi phenomena (from local inspection): `SV-G` (gender), `SV-P` (person), `SV-#` (number).

Re-download:

```bash
curl -L -o eng_data.tsv https://huggingface.co/datasets/jumelet/multiblimp/resolve/main/eng/data.tsv
curl -L -o hin_data.tsv https://huggingface.co/datasets/jumelet/multiblimp/resolve/main/hin/data.tsv
```

Used by `src/data_multiblimp.py` as the default agreement source.
