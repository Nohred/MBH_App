import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as nifti from 'nifti-reader-js';

// --- 1. Basic Three.js Setup ---
const canvas = document.getElementById('webgl-canvas');
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 300; // Moved camera back to see the whole scan

const renderer = new THREE.WebGLRenderer({ canvas: canvas });
renderer.setSize(window.innerWidth, window.innerHeight);

const controls = new OrbitControls(camera, renderer.domElement);



// We will store our volume mesh here
let volumeMesh;

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();


// --- 2. UI Elements ---
const scanInput = document.getElementById('upload-scan');
const renderBtn = document.getElementById('render-btn');
const statusLog = document.getElementById('status-log');
let selectedFile = null;

function updateStatus(message) {
  statusLog.innerText = `Status: ${message}`;
}

scanInput.addEventListener('change', (e) => {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    renderBtn.disabled = false;
    updateStatus("File selected. Ready to render.");
  }
});




// --- 3. Parsing and Rendering Logic ---
renderBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  
  updateStatus("Reading file into RAM...");
  const reader = new FileReader();
  
  reader.onload = function(event) {
    let data = event.target.result;
    
    updateStatus("Decompressing and parsing NIFTI...");
    if (nifti.isCompressed(data)) data = nifti.decompress(data);
    
    if (nifti.isNIFTI(data)) {
      const header = nifti.readHeader(data);
      const image = nifti.readImage(header, data);
      
      createVolumeRendering(header, image);
    } else {
      updateStatus("Error: Invalid NIFTI file.");
    }
  };
  
  reader.readAsArrayBuffer(selectedFile);
});





function createVolumeRendering(header, image) {
  updateStatus("Converting data for GPU...");

  // 1. Parse Data
  let typedData;
  if (header.datatypeCode === 2) typedData = new Uint8Array(image);
  else if (header.datatypeCode === 4) typedData = new Int16Array(image);
  else if (header.datatypeCode === 8) typedData = new Int32Array(image);
  else if (header.datatypeCode === 16) typedData = new Float32Array(image);
  else if (header.datatypeCode === 64) typedData = new Float64Array(image);
  else typedData = new Uint8Array(image);

  let absMin = Infinity, absMax = -Infinity;
  for (let i = 0; i < typedData.length; i++) {
    if (typedData[i] < absMin) absMin = typedData[i];
    if (typedData[i] > absMax) absMax = typedData[i];
  }

  const airValue = Math.max(typedData[0], typedData[typedData.length - 1]);
  let autoMin = airValue + 1;
  let autoMax = absMax;

  // CT SCAN DETECTOR 
  if (absMin <= -1000 && absMax > 1000) {
    autoMin = -500; // Start capturing at soft tissue/skin (-500 HU)
    autoMax = 1500; // Stop capping at hard bone (1500 HU)
  }

  // Map to the 3D Texture
  const volumeData = new Uint8Array(typedData.length);
  const range = autoMax - autoMin;
  
  for (let i = 0; i < typedData.length; i++) {
    let val = typedData[i];
    if (val <= autoMin) volumeData[i] = 0; // Transparent background air
    else if (val >= autoMax) volumeData[i] = 255; // Max density bone
    else volumeData[i] = ((val - autoMin) / range) * 255;
  }






  // 2. Create the 3D Texture
  const texture = new THREE.Data3DTexture(volumeData, header.dims[1], header.dims[2], header.dims[3]);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  
  // FIX: LinearFilter smoothly blends the voxels together!
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;


  // 3. Create the Cinematic Volume Material
  const volumeMaterial = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      map: { value: texture }
    },
    vertexShader: /* glsl */`
      out vec3 vOrigin;
      out vec3 vDirection;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
        vDirection = position - vOrigin;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */`
      precision highp float;
      precision highp sampler3D;
      
      uniform sampler3D map;
      in vec3 vOrigin;
      in vec3 vDirection;
      out vec4 color;

      vec2 hitBox(vec3 orig, vec3 dir) {
        const vec3 box_min = vec3(-0.5);
        const vec3 box_max = vec3(0.5);
        vec3 inv_dir = 1.0 / dir;
        vec3 tmin_tmp = (box_min - orig) * inv_dir;
        vec3 tmax_tmp = (box_max - orig) * inv_dir;
        vec3 tmin = min(tmin_tmp, tmax_tmp);
        vec3 tmax = max(tmin_tmp, tmax_tmp);
        float t0 = max(tmin.x, max(tmin.y, tmin.z));
        float t1 = min(tmax.x, min(tmax.y, tmax.z));
        return vec2(t0, t1);
      }

      // Calculates the 3D surface angle (gradient) for realistic lighting shadows
      vec3 getNormal(vec3 p) {
        float h = 0.005; 
        vec3 n;
        n.x = texture(map, p + vec3(h, 0, 0)).r - texture(map, p - vec3(h, 0, 0)).r;
        n.y = texture(map, p + vec3(0, h, 0)).r - texture(map, p - vec3(0, h, 0)).r;
        n.z = texture(map, p + vec3(0, 0, h)).r - texture(map, p - vec3(0, 0, h)).r;
        return normalize(n);
      }

      void main() {
        vec3 rayDir = normalize(vDirection);
        vec2 bounds = hitBox(vOrigin, rayDir);
        if (bounds.x > bounds.y) discard;
        bounds.x = max(bounds.x, 0.0);
        
        vec3 p = vOrigin + bounds.x * rayDir;
        float delta = 0.0025; 
        
        vec4 accum = vec4(0.0); // We will pile layers of color onto this
        
        // Simulates a lamp attached to the camera, shining slightly from above
        vec3 lightDir = normalize(-rayDir + vec3(0.0, 0.5, 0.0)); 
        
        for (float t = bounds.x; t < bounds.y; t += delta) {
          float val = texture(map, p + 0.5).r;
          
          if (val > 0.05) { // Skip empty air
            
            // 🎨 TRANSFER FUNCTION: Map density to colors and opacities
            vec4 voxelColor;
            if (val < 0.3) {
              // Skin/Soft Tissue: Pinkish, highly transparent
              voxelColor = vec4(0.9, 0.6, 0.5, val * 0.15); 
            } else if (val < 0.45) {
              // Muscle/Transition: darker, semi-transparent
              voxelColor = vec4(0.8, 0.3, 0.3, val * 0.4);
            } else {
              // Bone: Solid, opaque, off-white
              voxelColor = vec4(1.0, 0.95, 0.9, val * 2.0);
            }
            
            // 💡 LIGHTING: Calculate shadows based on the 3D surface
            vec3 normal = getNormal(p + 0.5);
            float diffuse = max(dot(normal, lightDir), 0.0);
            
            // Mix ambient light (0.4) with directional light (0.6)
            vec3 rgb = voxelColor.rgb * (diffuse * 0.6 + 0.4); 
            
            // Blend this voxel over the ones behind it (Front-to-back compositing)
            float alpha = clamp(voxelColor.a, 0.0, 1.0);
            accum.rgb += (1.0 - accum.a) * rgb * alpha;
            accum.a += (1.0 - accum.a) * alpha;
            
            // Stop calculating if the pixel is completely solid to save performance
            if (accum.a >= 0.98) break;
          }
          
          p += rayDir * delta;
        }
        
        if (accum.a == 0.0) discard;
        color = accum;
      }
    `,
    side: THREE.BackSide,
    transparent: true
  });




  
  if (volumeMesh) scene.remove(volumeMesh);

  // FIX: Calculate exact physical volume dimensions using NIFTI pixDims
  const xSize = header.dims[1] * header.pixDims[1];
  const ySize = header.dims[2] * header.pixDims[2];
  const zSize = header.dims[3] * header.pixDims[3];
  
  // Normalize sizes so the largest side is exactly 150 units in Three.js
  const maxSpace = Math.max(xSize, ySize, zSize);

  const geometry = new THREE.BoxGeometry(1, 1, 1);
  volumeMesh = new THREE.Mesh(geometry, volumeMaterial);
  
  // Apply the corrected physical proportions
  volumeMesh.scale.set(
    (xSize / maxSpace) * 150, 
    (ySize / maxSpace) * 150, 
    (zSize / maxSpace) * 150
  );
  
  scene.add(volumeMesh);
  updateStatus("Render complete! You can drag to rotate the volume.");
}


// Window resizing
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
