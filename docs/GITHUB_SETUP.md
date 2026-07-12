# GitHub Setup

## Create the repository

Create a new empty repository, for example:

```text
sambasportai-world-cup-2026
```

Do not initialize it with another README because this package already contains one.

## Push the project

```bash
git init
git add .
git commit -m "Initial SambaSportAI World Cup 2026 release"
git branch -M main
git remote add origin https://github.com/areyesan/sambasportai-world-cup-2026.git
git push -u origin main
```

## Enable GitHub Pages

The repository includes `.github/workflows/pages.yml`.

In GitHub:

1. Open **Settings**.
2. Open **Pages**.
3. Under **Build and deployment**, select **GitHub Actions**.
4. Push to `main`.

The `ui/` directory will be deployed as a static website.

## Recommended repository settings

- Add topics: `football`, `soccer`, `sports-analytics`, `machine-learning`, `world-cup`, `probabilistic-forecasting`.
- Enable Issues and Discussions if the project will be public.
- Protect the `main` branch and require the CI workflow to pass.
- Replace the restrictive `LICENSE` file if you decide to release the code under MIT, Apache-2.0, or another open-source license.
