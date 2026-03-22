import * as nifti from 'nifti-reader-js';
import * as THREE from 'three';

// Reads the raw file into an ArrayBuffer and decompresses it
export function readFileToNifti(file, callback, errorCallback) {
  const reader = new FileReader();
  reader.onload = function(event) {
    let data = event.target.result;
    if (nifti.isCompressed(data)) data = nifti.decompress(data);
    
    if (nifti.isNIFTI(data)) {
      const header = nifti.readHeader(data);
      const image = nifti.readImage(header, data);
      callback(header, image);
    } else {
      errorCallback("Error: Invalid NIFTI file.");
    }
  };
  reader.readAsArrayBuffer(file);
}

// Converts NIFTI image data into a typed array safely
export function getTypedData(header, image) {
  switch (header.datatypeCode) {
    case 2: return new Uint8Array(image);     // UINT8
    case 4: return new Int16Array(image);     // INT16
    case 8: return new Int32Array(image);     // INT32
    case 16: return new Float32Array(image);  // FLOAT32
    case 64: return new Float64Array(image);  // FLOAT64
    case 256: return new Int8Array(image);    // INT8  <-- Missing!
    case 512: return new Uint16Array(image);  // UINT16 <-- Missing!
    case 768: return new Uint32Array(image);  // UINT32 <-- Missing!
    default:
      console.warn("Unknown datatypeCode:", header.datatypeCode, "- defaulting to Uint8");
      return new Uint8Array(image);
  }
}


// Creates the Three.js Data3DTexture for the main brain scan
export function createMainTexture(header, typedData) {
  let absMin = Infinity, absMax = -Infinity;
  for (let i = 0; i < typedData.length; i++) {
    if (typedData[i] < absMin) absMin = typedData[i];
    if (typedData[i] > absMax) absMax = typedData[i];
  }

  const airValue = Math.max(typedData[0], typedData[typedData.length - 1]);
  let autoMin = airValue + 1;
  let autoMax = absMax;

  if (absMin <= -1000 && absMax > 1000) {
    autoMin = -500;
    autoMax = 1500;
  }

  const volumeData = new Uint8Array(typedData.length);
  const range = autoMax - autoMin;
  
  for (let i = 0; i < typedData.length; i++) {
    let val = typedData[i];
    if (val <= autoMin) volumeData[i] = 0;
    else if (val >= autoMax) volumeData[i] = 255;
    else volumeData[i] = ((val - autoMin) / range) * 255;
  }

  const texture = new THREE.Data3DTexture(volumeData, header.dims[1], header.dims[2], header.dims[3]);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;

  return texture;
}

// Creates the Three.js Data3DTexture specifically for the Mask
export function createMaskTexture(header, typedData) {
  const maskVolumeData = new Uint8Array(typedData.length);
  for (let i = 0; i < typedData.length; i++) {
    maskVolumeData[i] = typedData[i] > 0 ? 255 : 0;
  }

  const maskTexture = new THREE.Data3DTexture(maskVolumeData, header.dims[1], header.dims[2], header.dims[3]);
  maskTexture.format = THREE.RedFormat;
  maskTexture.type = THREE.UnsignedByteType;
  maskTexture.minFilter = THREE.NearestFilter; 
  maskTexture.magFilter = THREE.NearestFilter;
  maskTexture.unpackAlignment = 1;
  maskTexture.needsUpdate = true;

  return maskTexture;
}
