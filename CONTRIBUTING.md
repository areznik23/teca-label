# Contributing

```
pip install -e '.[postgres,openai]'
python -m pyflakes teca_label tests
python -m unittest discover -s tests
```

Tests make no model or database calls; both extras are needed only so their modules import.

The codebook file, the labels table, and the runs table are the contract, specified in
[FORMAT.md](FORMAT.md). A change to any of them is additive or bumps `schema`, and comes with
its FORMAT.md entry in the same commit. Everything else is implementation and can change.

Keep the library small. A feature earns its place when the scheduled runner uses it or the
README shows it; anything else lives in your own project on top of the public API.
