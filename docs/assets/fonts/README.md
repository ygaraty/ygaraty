# Documentation fonts

Self-hosted Latin subsets of Fraunces (roman and italic), IBM Plex Sans, and
IBM Plex Mono (400 and 500), from Google Fonts. Distributed under SIL Open
Font License 1.1; see the accompanying license files. No external font service
is required by the documentation.

MyST copies the stylesheet without resolving its font URLs. Font data is embedded
in `../style.css`; regenerate it after changing fonts with:

```sh
python3 docs/assets/fonts/embed.py
```
