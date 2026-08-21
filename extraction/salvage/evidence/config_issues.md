# Two things the salvage turned up in `config.py`

Neither was changed. Both are one-line decisions that belong to you.

## 1. `LOG_SPECS["http"].expected_sample_rows` is off by one

```python
"http": LogSpec(..., expected_sample_rows=967_806),
```

The sample's `http.csv` has **967,806 lines including the header**, so
**967,805 data rows**. The constant counted the header line. Every other log's
count is right: logon 67,953 (35,122 logons + 32,831 logoffs), device 26,342,
file 26,222, email 89,577 — all reproduce exactly.

Consequence: `verify_setup.py --full` will report a mismatch on http and only
on http. The fix is `967_805`.

## 2. `JOB_SEARCH_DOMAINS` misses two domains that fire on the real data

Measured over the whole 30-user sample `http.csv` — see `domain_hits.csv`:

| domain | list | in `config.py`? | hits |
|---|---|---|---|
| craigslist.org | job | **no** | 3,855 |
| careerbuilder.com | job | yes | 3,347 |
| monster.com | job | yes | 2,927 |
| jobhuntersbible.com | job | yes | 2,185 |
| job-hunt.org | job | yes | 2,185 |
| indeed.com | job | yes | 2,183 |
| simplyhired.com | job | yes | 2,150 |
| aol.com/jobs | job | **no** | 2,088 |
| linkedin.com | job | excluded on purpose | 40,446 |
| mediafire.com | cloud | yes | 340 |
| dropbox.com | cloud | yes | 114 |
| wikileaks.org | cloud | yes | 7 |
| 4shared.com | cloud | yes | 1 |

Nothing is broken — every domain currently in the list fires, so feature 16 and
feature 17 are both live. But roughly **6,000 job-search visits in a 30-user
sample are currently invisible to feature 16**, and scenario 2 is thirty
insiders whose entire signature is job hunting.

The tension is real and worth stating in the write-up rather than resolving
quietly: the innersight2 Slice 2 list was measured off the data and included
craigslist and aol.com/jobs; the STAGE0 brief's list is the one in `config.py`
and includes job-hunt.org and simplyhired instead. Excluding linkedin.com is
correct either way — at 40,446 hits it is background browsing and would drown
the feature.

Two caveats if you do add them:

* `aol.com/jobs` is a path, not a host. The matcher is substring-against-
  lowercased-URL, so it works — but it would not under host equality.
* `craigslist.org` is not purely a job site. Counting all of it as job-search
  is a defensible simplification, not a measurement; say so if asked.

The cloud list is thin on purpose (462 hits total in the sample). That is the
dataset, not the code — scenario 1 and 3 exfiltrate over USB, not the web.
