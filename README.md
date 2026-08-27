# MBH_App

Application web to visualize and classify hemorrhages in CT images (NIfTI volumes).

Part of the MICCAI brain hemorrhage segmentation/classification project. This app renders CT volumes in the browser and overlays voxel-level segmentation masks / hemorrhage type labels for inspection.

## Requirements

- **Node.js**: `20.19+` or `22.12+` (Vite 6+ requires one of these; Node 18 is **not** supported anymore).
- **npm**: bundled with Node.js (v9+ recommended).
- A modern browser with WebGL support (for volume rendering).

### Recommended: manage Node.js with nvm

If you don't have a compatible Node version, install [nvm](https://github.com/nvm-sh/nvm) and use it to install/switch versions instead of relying on your OS package manager:

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
source ~/.bashrc   # or ~/.zshrc

# Install the latest LTS Node version
nvm install --lts
nvm alias default 'lts/*'

# Verify
node -v   # should print v22.x or v24.x
```

This repo includes an `.nvmrc` file, so once nvm is installed you can simply run `nvm use` inside the project folder to switch to the correct version automatically.

## Setup

Clone the repo and install dependencies:

```bash
git clone <repo-url>
cd MBH_App
npm install
```

## Running the app (development)

```bash
npm run dev
```

This starts the Vite dev server. Open the printed local URL (usually `http://localhost:5173`) in your browser.

## Building for production

```bash
npm run build
npm run preview   # to test the production build locally
```

## Project structure

```
.
├── code/                    # Data science / ML notebooks (not part of the web app)
│   ├── data_cleaning.ipynb  # Preprocessing of CT/NIfTI data and masks
│   ├── models_training.ipynb# Segmentation/classification model training (PyTorch/MONAI)
│   └── prueba.py
├── index.html               # Vite entry point
├── src/
│   ├── main.js               # App entry point
│   ├── niftiParser.js         # NIfTI file parsing utilities
│   ├── volumeRender.js        # 3D/2D volume rendering logic
│   └── style.css
├── package.json
└── README.md
```

> Note: the `code/` directory contains the Python/Jupyter side of the project (data cleaning and model training for hemorrhage segmentation/classification) and is independent from this Node/Vite visualization app. It requires its own Python environment (PyTorch, MONAI, nibabel, etc.), not `npm`.

## Troubleshooting

- **`sh: vite: not found`**: dependencies were never installed. Run `npm install` inside the project folder.
- **`Vite requires Node.js version 20.19+ or 22.12+`**: your Node.js version is too old. Upgrade using nvm (see above).
- If issues persist after upgrading Node, try a clean reinstall:
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```

## License

See [LICENSE](./LICENSE).
