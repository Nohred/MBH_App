import * as THREE from 'three';

export function createVolumeMaterial(mainTexture) {
  // Dummy texture to prevent shader crashes before mask is loaded
  const dummyData = new Uint8Array([0]);
  const dummyTexture = new THREE.Data3DTexture(dummyData, 1, 1, 1);
  dummyTexture.format = THREE.RedFormat;
  dummyTexture.type = THREE.UnsignedByteType;
  dummyTexture.needsUpdate = true;

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      map: { value: mainTexture },
      maskMap: { value: dummyTexture },
      uHasMask: { value: false }
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
      uniform sampler3D maskMap;
      uniform bool uHasMask;
      
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
        vec4 accum = vec4(0.0);
        vec3 lightDir = normalize(-rayDir + vec3(0.0, 0.5, 0.0)); 
        
        for (float t = bounds.x; t < bounds.y; t += delta) {
          float val = texture(map, p + 0.5).r;
          float maskVal = uHasMask ? texture(maskMap, p + 0.5).r : 0.0;
          
          if (val > 0.05 || maskVal > 0.5) { 
            vec3 rgb;
            float voxelAlpha;

            if (maskVal > 0.5) {
              // Bright Orange/Red Mask
              rgb = vec3(1.0, 0.3, 0.0);
              voxelAlpha = 0.5;
            } else {
              // Transparent Grayscale Reference
              vec3 baseColor = vec3(1.0);
              voxelAlpha = val * 0.06;
              vec3 normal = getNormal(p + 0.5);
              float diffuse = max(dot(normal, lightDir), 0.0);
              rgb = baseColor * (diffuse * 0.7 + 0.3);
            }
            
            accum.rgb += (1.0 - accum.a) * rgb * voxelAlpha;
            accum.a += (1.0 - accum.a) * voxelAlpha;
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
}
