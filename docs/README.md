# Public documentation

The user guide starts at [index.md](index.md). This is a standalone MyST project;
`myst.yml` defines the book navigation and theme. Edit guides here and keep
internal engineering notes in the relevant source-folder READMEs.

```sh
npm install --global mystmd@1.14.0
cd docs
myst start
BASE_URL=/ygaraty myst build --html --strict
```

GitHub Actions validates pull requests and publishes main to
https://ygaraty.github.io/ygaraty/. Generated `_build/` files are ignored.
The project owns its brand assets and licensed fonts; it has no cloud dependency.
