import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { readFileToNifti, getTypedData, createMainTexture, createMaskTexture } from './niftiParser.js';
import { createVolumeMaterial, setSliceType, updateSlicePosition } from './volumeRender.js';
import { createColorLegend, renderSlice2D } from './sliceRender2D.js';

// --- 1. Basic Three.js Setup ---
const canvas = document.getElementById('webgl-canvas');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 300;

let renderer = null;
let controls = null;
try {
  renderer = new THREE.WebGLRenderer({ canvas: canvas });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  controls = new OrbitControls(camera, renderer.domElement);
} catch (error) {
  // La interfaz sigue disponible para mostrar el diagnóstico si el navegador no ofrece WebGL.
  console.error('No se pudo inicializar WebGL:', error);
}

let volumeMesh = null; // Store mesh globally to access it later
const appState = {
  screenId: 'home-screen',
  viewMode: 'volume',
};

function animate() {
  requestAnimationFrame(animate);
  const isViewerScreen = appState.screenId === 'visualize-screen' || appState.screenId === 'segment-screen';
  if (!renderer || !controls || appState.viewMode !== 'volume' || !isViewerScreen) return;
  controls.update();
  renderer.render(scene, camera);
}
animate();

// --- 2. Screen navigation and shared view controls ---
const canvasElement = document.getElementById('webgl-canvas');
const sliceCanvasElement = document.getElementById('slice-canvas');
const screens = document.querySelectorAll('.screen');
const colorLegend = createColorLegend();
document.body.appendChild(colorLegend);
let volumeData = null;

function getSliceAxis(mode) {
  return mode === 'sagittal' ? 'sagittal' : mode;
}

function getSliceCount(mode, header) {
  const dimensionByAxis = { axial: 3, coronal: 2, sagittal: 1 };
  return Number(header?.dims?.[dimensionByAxis[mode]] || 1);
}

function renderActiveSlice(activeControls, mode) {
  if (!volumeData || !['axial', 'coronal', 'sagittal'].includes(mode)) return;
  const slider = activeControls.querySelector('[data-slice-slider]');
  const windowLevel = activeControls.querySelector('[data-window-level]');
  const windowWidth = activeControls.querySelector('[data-window-width]');
  renderSlice2D(sliceCanvasElement, volumeData.header, volumeData.typedData, volumeData.maskTypedData, {
    axis: getSliceAxis(mode),
    sliceIndex: Number(slider.value),
    // TODO: connect these controls to live redraws when interactive WL/WW is enabled.
    windowLevel: Number(windowLevel.value) || 40,
    windowWidth: Number(windowWidth.value) || 120,
  });
}

function syncActiveView() {
  const activeControls = document.querySelector('.screen:not(.hidden) [data-view-controls]');
  const activeMode = activeControls?.querySelector('[data-view-mode]');
  const sliceSlider = activeControls?.querySelector('[data-slice-slider]');
  const mode = activeMode?.value || 'volume';
  const isSliceMode = ['axial', 'coronal', 'sagittal'].includes(mode);
  appState.viewMode = mode;

  setSliceType(mode, volumeMesh);
  if (sliceSlider) {
    sliceSlider.disabled = mode === 'volume';
    activeControls.classList.toggle('slice-hidden', mode === 'volume');
    if (isSliceMode && volumeData) {
      const previousFraction = Number(sliceSlider.max) > 0
        ? Number(sliceSlider.value) / Number(sliceSlider.max)
        : 0.5;
      sliceSlider.max = String(Math.max(0, getSliceCount(mode, volumeData.header) - 1));
      sliceSlider.value = String(Math.round(previousFraction * Number(sliceSlider.max)));
    }
  }

  const currentScreen = document.querySelector('.screen:not(.hidden)');
  const isHome = currentScreen?.id === 'home-screen';
  const showWebgl = !isHome && Boolean(volumeMesh) && !isSliceMode;
  const showSlice = !isHome && isSliceMode && Boolean(volumeData);
  canvasElement.classList.toggle('hidden', !showWebgl);
  sliceCanvasElement.classList.toggle('hidden', !showSlice);
  colorLegend.classList.toggle('hidden', isHome);
  if (isSliceMode) renderActiveSlice(activeControls, mode);
}

function showScreen(screenId) {
  screens.forEach((screen) => screen.classList.toggle('hidden', screen.id !== screenId));
  appState.screenId = screenId;
  syncActiveView();
}

document.querySelectorAll('[data-go-screen]').forEach((button) => {
  button.addEventListener('click', () => showScreen(button.dataset.goScreen));
});

function setupViewControls() {
  document.querySelectorAll('[data-view-controls]').forEach((controls) => {
    const modeSelect = controls.querySelector('[data-view-mode]');
    const sliceSlider = controls.querySelector('[data-slice-slider]');

    const applyView = () => {
      syncActiveView();
    };

    modeSelect.addEventListener('change', () => {
      if (controls.closest('.screen')?.id === appState.screenId) applyView();
    });
    sliceSlider.addEventListener('input', () => {
      if (['axial', 'coronal', 'sagittal'].includes(modeSelect.value)) {
        renderActiveSlice(controls, modeSelect.value);
      } else {
        updateSlicePosition(Number(sliceSlider.value) / 100, volumeMesh);
      }
    });
    controls.querySelectorAll('[data-window-level], [data-window-width]').forEach((input) => {
      input.addEventListener('input', () => {
        if (controls.closest('.screen')?.id === appState.screenId &&
            ['axial', 'coronal', 'sagittal'].includes(modeSelect.value)) {
          renderActiveSlice(controls, modeSelect.value);
        }
      });
    });
    applyView();
  });
}

setupViewControls();
showScreen('home-screen');

// --- 3. UI Elements ---
const scanInput = document.getElementById('upload-scan');
const renderMainBtn = document.getElementById('render-main-btn');
const maskInput = document.getElementById('upload-mask');
const renderMaskBtn = document.getElementById('render-mask-btn');
const statusLog = document.getElementById('status-log');
const segmentScanInput = document.getElementById('segment-scan-input');
const segmentBtn = document.getElementById('segment-btn');
const segmentStatusLog = document.getElementById('segment-status-log');

let selectedVisualizeFile = null;
let selectedSegmentFile = null;
let selectedMaskFile = null;

function updateStatus(message, target = statusLog) {
  target.innerText = `Status: ${message}`;
}

if (!renderer) {
  const webglMessage = 'WebGL no está disponible en este navegador; habilítalo para renderizar el volumen.';
  updateStatus(webglMessage, statusLog);
  updateStatus(webglMessage, segmentStatusLog);
}

// Handle file selections
scanInput.addEventListener('change', (e) => {
  selectedVisualizeFile = e.target.files[0];
  if (selectedVisualizeFile) renderMainBtn.disabled = false;
});

segmentScanInput.addEventListener('change', (e) => {
  selectedSegmentFile = e.target.files[0];
  if (selectedSegmentFile) segmentBtn.disabled = false;
});

maskInput.addEventListener('change', (e) => {
  selectedMaskFile = e.target.files[0];
  if (selectedMaskFile) renderMaskBtn.disabled = false;
});

// NIfTI loading and volume rendering are operational; inference remains a visual placeholder.
function renderScan(file, targetStatus, onRendered, enableMaskInput = false) {
  if (!renderer) {
    updateStatus('No se puede renderizar: WebGL no está disponible.', targetStatus);
    return;
  }
  updateStatus('Leyendo NIfTI principal...', targetStatus);

  readFileToNifti(file, (header, image) => {
    updateStatus('Procesando NIfTI para GPU...', targetStatus);
    const typedData = getTypedData(header, image);
    volumeData = { header, typedData, maskTypedData: null };
    const mainTexture = createMainTexture(header, typedData);

    // Create Mesh
    const material = createVolumeMaterial(mainTexture);
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    
    if (volumeMesh) scene.remove(volumeMesh);
    volumeMesh = new THREE.Mesh(geometry, material);

    // Scale physically
    const xSize = header.dims[1] * header.pixDims[1];
    const ySize = header.dims[2] * header.pixDims[2];
    const zSize = header.dims[3] * header.pixDims[3];
    const maxSpace = Math.max(xSize, ySize, zSize);
    volumeMesh.scale.set((xSize / maxSpace) * 150, (ySize / maxSpace) * 150, (zSize / maxSpace) * 150);

    scene.add(volumeMesh);
    canvasElement.classList.remove('hidden');
    
    syncActiveView();
    if (enableMaskInput) maskInput.disabled = false;
    onRendered();
  }, (message) => updateStatus(message, targetStatus));
}

// --- 4. Render Main Scan ---
renderMainBtn.addEventListener('click', () => {
  if (!selectedVisualizeFile) return;
  renderScan(selectedVisualizeFile, statusLog, () => {
    updateStatus('Estudio renderizado. Ya puedes cargar una máscara.', statusLog);
  }, true);
});

segmentBtn.addEventListener('click', () => {
  if (!selectedSegmentFile) return;
  const selectedModel = document.getElementById('model-select')?.value;
  if (selectedModel === 'model-c') {
    updateStatus('El Modelo C todavía no está implementado.', segmentStatusLog);
    return;
  }

  const modelId = selectedModel === 'model-a' ? 'model_a' : 'model_b';
  segmentBtn.disabled = true;
  renderScan(selectedSegmentFile, segmentStatusLog, async () => {
    updateStatus('Ejecutando segmentación, puede tardar unos segundos...', segmentStatusLog);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/predict', {
        method: 'POST',
        body: (() => {
          const formData = new FormData();
          formData.append('file', selectedSegmentFile);
          formData.append('model_id', modelId);
          return formData;
        })(),
      });

      if (!response.ok) {
        let errorMessage = `Error del backend (${response.status}).`;
        try {
          const errorBody = await response.json();
          errorMessage = errorBody.error?.message || errorMessage;
        } catch {
          // Conserva el mensaje HTTP si la respuesta no es JSON.
        }
        throw new Error(errorMessage);
      }

      const maskFile = new File([await response.blob()], `mask_${modelId}.nii.gz`, {
        type: 'application/gzip',
      });
      readFileToNifti(maskFile, (maskHeader, maskImage) => {
        const maskDimensions = [maskHeader.dims[1], maskHeader.dims[2], maskHeader.dims[3]];
        const volumeDimensions = [volumeData.header.dims[1], volumeData.header.dims[2], volumeData.header.dims[3]];
        if (maskDimensions.some((dimension, index) => Number(dimension) !== Number(volumeDimensions[index]))) {
          segmentBtn.disabled = false;
          updateStatus(`La máscara del ${modelId} no coincide con las dimensiones del volumen original.`, segmentStatusLog);
          return;
        }

        const maskTypedData = getTypedData(maskHeader, maskImage);
        const maskTexture = createMaskTexture(maskHeader, maskTypedData);
        volumeData.maskTypedData = maskTypedData;
        volumeMesh.material.uniforms.maskMap.value = maskTexture;
        volumeMesh.material.uniforms.uHasMask.value = true;
        syncActiveView();
        segmentBtn.disabled = false;
        updateStatus('Segmentación completada. La máscara está disponible en los cortes 2D.', segmentStatusLog);
      }, (message) => {
        segmentBtn.disabled = false;
        updateStatus(`No se pudo leer la máscara devuelta: ${message}`, segmentStatusLog);
      });
    } catch (error) {
      segmentBtn.disabled = false;
      updateStatus(`No se pudo completar la segmentación: ${error.message}`, segmentStatusLog);
    }
  });
});

// --- 5. Apply Mask ---
renderMaskBtn.addEventListener('click', () => {
  if (!selectedMaskFile || !volumeMesh) return;
  updateStatus('Leyendo máscara...', statusLog);

  readFileToNifti(selectedMaskFile, (header, image) => {
    updateStatus('Aplicando máscara a la escena 3D...', statusLog);
    const typedData = getTypedData(header, image);
    const maskTexture = createMaskTexture(header, typedData);
    if (volumeData) volumeData.maskTypedData = typedData;

    // Inject into the running shader
    volumeMesh.material.uniforms.maskMap.value = maskTexture;
    volumeMesh.material.uniforms.uHasMask.value = true;
    
    syncActiveView();
    updateStatus('Máscara aplicada. La hemorragia aparece por clase en los cortes 2D.', statusLog);
  }, (message) => updateStatus(message, statusLog));
});

// --- 6. Resize Handler ---
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  if (renderer) renderer.setSize(window.innerWidth, window.innerHeight);
  syncActiveView();
});
