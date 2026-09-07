export const CLASS_COLORS = {
  1: [1.0, 0.15, 0.15],
  2: [0.15, 1.0, 0.15],
  3: [0.15, 0.45, 1.0],
  4: [1.0, 1.0, 0.0],
  5: [0.0, 1.0, 1.0],
};

const CLASS_LABELS = {
  1: 'Epidural',
  2: 'Subdural',
  3: 'Subarachnoid',
  4: 'Intraventricular',
  5: 'Intracerebral',
};

export function createColorLegend(documentRef = document) {
  const legend = documentRef.createElement('aside');
  legend.className = 'color-legend';
  legend.setAttribute('aria-label', 'Tipos de hemorragia');

  const title = documentRef.createElement('h3');
  title.textContent = 'Tipos de hemorragia';
  legend.appendChild(title);

  Object.entries(CLASS_COLORS).forEach(([classValue, color]) => {
    const row = documentRef.createElement('div');
    row.className = 'color-legend-row';

    const swatch = documentRef.createElement('span');
    swatch.className = 'color-legend-swatch';
    swatch.style.backgroundColor = `rgb(${color.map((channel) => Math.round(channel * 255)).join(', ')})`;
    swatch.setAttribute('aria-hidden', 'true');

    const label = documentRef.createElement('span');
    label.textContent = CLASS_LABELS[classValue];

    row.append(swatch, label);
    legend.appendChild(row);
  });

  return legend;
}

const AXES = {
  axial: { width: 0, height: 1, slice: 2 },
  coronal: { width: 0, height: 2, slice: 1 },
  sagittal: { width: 1, height: 2, slice: 0 },
  sagital: { width: 1, height: 2, slice: 0 },
};

function getVoxelIndex(x, y, z, dimensions) {
  return x + dimensions[0] * (y + dimensions[1] * z);
}

function getScaledValue(value, header) {
  const slope = Number(header.scl_slope) || 1;
  const intercept = Number(header.scl_inter) || 0;
  return value * slope + intercept;
}

export function renderSlice2D(
  canvas2D,
  header,
  imageTypedArray,
  maskTypedArray,
  {
    axis = 'axial',
    sliceIndex = 0,
    windowLevel = 40,
    windowWidth = 120,
  } = {},
) {
  if (!canvas2D || !header || !imageTypedArray) return;

  const dimensions = [header.dims[1], header.dims[2], header.dims[3]].map(Number);
  const axisConfig = AXES[axis] || AXES.axial;
  const width = dimensions[axisConfig.width];
  const height = dimensions[axisConfig.height];
  const pixelWidth = Number(header.pixDims?.[axisConfig.width + 1]) || 1;
  const pixelHeight = Number(header.pixDims?.[axisConfig.height + 1]) || 1;
  const maxSlice = dimensions[axisConfig.slice] - 1;
  const safeSlice = Math.max(0, Math.min(maxSlice, Math.round(Number(sliceIndex) || 0)));
  const safeWindowWidth = Math.max(1, Number(windowWidth) || 120);
  const windowMin = (Number(windowLevel) || 40) - safeWindowWidth / 2;
  const windowMax = (Number(windowLevel) || 40) + safeWindowWidth / 2;
  const imageData = new ImageData(width, height);

  canvas2D.width = width;
  canvas2D.height = height;
  const displayScale = Math.min(
    (window.innerWidth * 0.9) / (width * pixelWidth),
    (window.innerHeight * 0.9) / (height * pixelHeight),
  );
  const physicalWidth = width * pixelWidth;
  const physicalHeight = height * pixelHeight;
  canvas2D.style.width = `${Math.max(1, physicalWidth * displayScale)}px`;
  canvas2D.style.height = `${Math.max(1, physicalHeight * displayScale)}px`;
  canvas2D.style.aspectRatio = `${physicalWidth} / ${physicalHeight}`;

  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const coordinates = [0, 0, 0];
      coordinates[axisConfig.width] = column;
      coordinates[axisConfig.height] = row;
      coordinates[axisConfig.slice] = safeSlice;

      const voxelIndex = getVoxelIndex(coordinates[0], coordinates[1], coordinates[2], dimensions);
      const normalized = Math.max(0, Math.min(1,
        (getScaledValue(imageTypedArray[voxelIndex], header) - windowMin) /
        (windowMax - windowMin),
      ));
      const maskValue = maskTypedArray ? Math.round(maskTypedArray[voxelIndex]) : 0;
      const classColor = CLASS_COLORS[maskValue];
      const color = classColor
        ? classColor.map((channel) => normalized * 0.6 + channel * 0.4)
        : [normalized, normalized, normalized];
      const pixelOffset = (row * width + column) * 4;

      imageData.data[pixelOffset] = Math.round(color[0] * 255);
      imageData.data[pixelOffset + 1] = Math.round(color[1] * 255);
      imageData.data[pixelOffset + 2] = Math.round(color[2] * 255);
      imageData.data[pixelOffset + 3] = 255;
    }
  }

  const context = canvas2D.getContext('2d');
  context.putImageData(imageData, 0, 0);
}

// TODO: support dynamic colors when models expose more than the five current classes.
// TODO: connect these defaults to interactive window-level/window-width controls.
