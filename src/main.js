import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { readFileToNifti, getTypedData, createMainTexture, createMaskTexture } from './niftiParser.js';
import { createVolumeMaterial } from './volumeRender.js';

// --- 1. Basic Three.js Setup ---
const canvas = document.getElementById('webgl-canvas');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 300;

const renderer = new THREE.WebGLRenderer({ canvas: canvas });
renderer.setSize(window.innerWidth, window.innerHeight);

const controls = new OrbitControls(camera, renderer.domElement);

let volumeMesh = null; // Store mesh globally to access it later

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// --- 2. UI Elements ---
const scanInput = document.getElementById('upload-scan');
const renderMainBtn = document.getElementById('render-main-btn');
const maskInput = document.getElementById('upload-mask');
const renderMaskBtn = document.getElementById('render-mask-btn');
const statusLog = document.getElementById('status-log');

let selectedMainFile = null;
let selectedMaskFile = null;

function updateStatus(message) {
  statusLog.innerText = `Status: ${message}`;
}

// Handle file selections
scanInput.addEventListener('change', (e) => {
  selectedMainFile = e.target.files[0];
  if (selectedMainFile) renderMainBtn.disabled = false;
});

maskInput.addEventListener('change', (e) => {
  selectedMaskFile = e.target.files[0];
  if (selectedMaskFile) renderMaskBtn.disabled = false;
});

// --- 3. Render Main Scan ---
renderMainBtn.addEventListener('click', () => {
  if (!selectedMainFile) return;
  updateStatus("Reading Main Scan...");

  readFileToNifti(selectedMainFile, (header, image) => {
    updateStatus("Processing Main Scan for GPU...");
    const typedData = getTypedData(header, image);
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
    
    // Unlock Mask Features
    maskInput.disabled = false;
    updateStatus("Main Scan rendered! You can now upload a mask.");
  }, updateStatus);
});

// --- 4. Apply Mask ---
renderMaskBtn.addEventListener('click', () => {
  if (!selectedMaskFile || !volumeMesh) return;
  updateStatus("Reading Mask file...");

  readFileToNifti(selectedMaskFile, (header, image) => {
    updateStatus("Applying Mask to 3D Scene...");
    const typedData = getTypedData(header, image);
    const maskTexture = createMaskTexture(header, typedData);

    // Inject into the running shader
    volumeMesh.material.uniforms.maskMap.value = maskTexture;
    volumeMesh.material.uniforms.uHasMask.value = true;
    
    updateStatus("Mask applied! Look for the orange/red hemorrhage.");
  }, updateStatus);
});

// --- 5. Resize Handler ---
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
