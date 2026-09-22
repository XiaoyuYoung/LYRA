# LYRA project page

This directory contains the static GitHub Pages project site for LYRA. It adapts the [Nerfies project-page template](https://github.com/nerfies/nerfies.github.io) and uses the template under the [Creative Commons Attribution-ShareAlike 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

## Publish with GitHub Pages

1. Push the `docs/` directory to the repository's default branch.
2. Open **Settings -> Pages** in GitHub.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the default branch and the `/docs` folder, then save.

GitHub will display the final public URL after the first deployment completes.

## Local preview

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000/`.
