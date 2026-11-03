# Data provenance

The team obtained both raw datasets directly from the UCI Machine Learning
Repository. Original names files are included with the data. The original
download date was not recorded and is not inferred from local timestamps.

| Dataset | UCI page and ID | DOI | Attribution |
| --- | --- | --- | --- |
| Adult | https://archive.ics.uci.edu/dataset/2/adult (ID 2) | https://doi.org/10.24432/C5XW20 | Barry Becker and Ronny Kohavi (1996); data extracted from the 1994 Census database by Barry Becker |
| Mushroom | https://archive.ics.uci.edu/dataset/73/mushroom (ID 73) | https://doi.org/10.24432/C5959T | Jeff Schlimmer, donor in 1987; underlying source The Audubon Society Field Guide to North American Mushrooms (1981) |

The recorded project provenance review identifies CC BY 4.0 for both UCI
datasets: https://creativecommons.org/licenses/by/4.0/. Credit the providers and
repository, link the license and identify any preprocessing/formatting changes.
This data license is not a license grant for all team-authored code or slides.
No new live licensing check is claimed during packaging.

The raw file hashes match the official-archive hashes recorded in the existing
project provenance review:

```text
5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d  adult.data
a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05  adult.test
e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e  agaricus-lepiota.data
```

Adult contains 48,842 records in the combined original files, 14 input
attributes and the income label (`<=50K` / `>50K`). Mushroom contains 8,124
records, 22 input attributes in the raw schema and the class label (`e` edible,
`p` poisonous). Its modeling CSV removes the constant veil-type attribute,
leaving 21 inputs. Both raw datasets use `?` for missing values. See the saved
schema/missingness tables for the audit and the notes on Mushroom's actual
modeling treatment in KNOWN_ISSUES.md.
