import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { getClamp } from '../../utils/clamp.ts';
import { getBorderInterpolation } from '../../utils/interpolateBorder.ts';
import {
  computeConvolutionValue,
  rawDirectConvolution,
} from '../convolution.ts';

describe('convolution functions', () => {
  it('separable convolution compared to opencv', () => {
    const img = testUtils.load('opencv/test.png');
    const convoluted = img.separableConvolution(
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6, -0.3, -0.4],
      {
        borderType: 'reflect',
      },
    );

    const expected = testUtils.load('opencv/testConvolution.png');

    expect(convoluted).toMatchImage(expected);
  });

  it('direct convolution compared to opencv', () => {
    const img = testUtils.load('opencv/test.png');

    const kernelY = Matrix.columnVector([0.4, 0.5, 0.6, -0.3, -0.4]);
    const kernelX = Matrix.rowVector([0.1, 0.2, 0.3]);
    const kernel = kernelY.mmul(kernelX).to2DArray();

    const convoluted = img.directConvolution(kernel, {
      borderType: 'reflect',
    });

    const expected = testUtils.load('opencv/testConvolution.png');

    expect(convoluted).toMatchImage(expected);
  });

  it('normalized separated convolution should normalize as for direct convolution', () => {
    const img = testUtils.load('opencv/test.png');

    const kernelY = Matrix.columnVector([1, 0, 1]);
    const kernelX = Matrix.rowVector([2, 1, 2]);
    const kernel = kernelY.mmul(kernelX);

    const normalizedKernel = kernel.mul(1 / kernel.sum());

    const img1 = img.directConvolution(normalizedKernel.to2DArray());
    const img2 = img.separableConvolution(
      kernelX.to1DArray(),
      kernelY.to1DArray(),
      {
        normalize: true,
      },
    );

    expect(img1).toMatchImage(img2);
  });

  it('normalized direct convolution should divide the kernel by its total weight', () => {
    const image = testUtils.createGreyImage([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const kernel = [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ];

    const convoluted = image.directConvolution(kernel, {
      borderType: 'constant',
      borderValue: 0,
      normalize: true,
    });

    expect(convoluted).toMatchImageData([
      [1, 2, 2],
      [3, 5, 4],
      [3, 4, 3],
    ]);
    // top-left, middle-right, bottom-right and center
    expect(convoluted.getValue(0, 0, 0)).toBe(1);
    expect(convoluted.getValue(2, 1, 0)).toBe(4);
    expect(convoluted.getValue(2, 2, 0)).toBe(3);
    expect(convoluted.getValue(1, 1, 0)).toBe(5);
  });

  it('normalized direct and separable convolutions should give the same values', () => {
    const image = testUtils.createGreyImage([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    const onesKernel = [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ];
    const gaussianKernel = [
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1],
    ];

    const directOnes = image.directConvolution(onesKernel, {
      borderType: 'constant',
      borderValue: 0,
      normalize: true,
    });
    const separableOnes = image.separableConvolution([1, 1, 1], [1, 1, 1], {
      borderType: 'constant',
      borderValue: 0,
      normalize: true,
    });

    expect(directOnes).toMatchImageData([
      [1, 2, 2],
      [3, 5, 4],
      [3, 4, 3],
    ]);
    expect(separableOnes).toMatchImage(directOnes);

    const directGaussian = image.directConvolution(gaussianKernel, {
      borderType: 'constant',
      borderValue: 0,
      normalize: true,
    });
    const separableGaussian = image.separableConvolution([1, 2, 1], [1, 2, 1], {
      borderType: 'constant',
      borderValue: 0,
      normalize: true,
    });

    expect(directGaussian).toMatchImageData([
      [1, 2, 2],
      [3, 5, 4],
      [4, 5, 4],
    ]);
    expect(separableGaussian).toMatchImage(directGaussian);
    // top-left, middle-right, bottom-right and center
    expect(directGaussian.getValue(0, 0, 0)).toBe(1);
    expect(directGaussian.getValue(2, 1, 0)).toBe(4);
    expect(directGaussian.getValue(2, 2, 0)).toBe(4);
    expect(directGaussian.getValue(1, 1, 0)).toBe(5);
  });

  it('unnormalized convolutions should keep the raw weighted sums', () => {
    const image = testUtils.createGreyImage([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    const onesKernel = [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ];
    const gaussianKernel = [
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1],
    ];

    const directOnes = image.directConvolution(onesKernel, {
      borderType: 'constant',
      borderValue: 0,
    });
    const separableOnes = image.separableConvolution([1, 1, 1], [1, 1, 1], {
      borderType: 'constant',
      borderValue: 0,
    });

    expect(directOnes).toMatchImageData([
      [12, 21, 16],
      [27, 45, 33],
      [24, 39, 28],
    ]);
    expect(separableOnes).toMatchImage(directOnes);
    expect(directOnes.getValue(2, 2, 0)).toBe(28);

    const directGaussian = image.directConvolution(gaussianKernel, {
      borderType: 'constant',
      borderValue: 0,
    });
    const separableGaussian = image.separableConvolution([1, 2, 1], [1, 2, 1], {
      borderType: 'constant',
      borderValue: 0,
    });

    expect(directGaussian).toMatchImageData([
      [21, 36, 33],
      [52, 80, 68],
      [57, 84, 69],
    ]);
    expect(separableGaussian).toMatchImage(directGaussian);
    expect(directGaussian.getValue(2, 2, 0)).toBe(69);
  });

  it('normalizing a kernel with a negative total weight should throw', () => {
    const image = testUtils.createGreyImage([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const kernel = [
      [-1, -1, -1],
      [-1, -1, -1],
      [-1, -1, -1],
    ];

    expect(() => image.directConvolution(kernel, { normalize: true })).toThrow(
      'this kernel cannot be normalized',
    );
  });
});

describe('computeConvolutionValue', () => {
  it('round and clamp', () => {
    const image = testUtils.createGreyImage([
      [1, 1, 20],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    const kernel = [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ];
    const clamp = getClamp(image);
    const interpolateBorder = getBorderInterpolation('reflect101', [0]);

    expect(
      computeConvolutionValue(1, 1, 0, image, kernel, interpolateBorder, {
        clamp,
      }),
    ).toBe(28);
  });

  it('round and clamp with negative kernel values', () => {
    const image = testUtils.createGreyImage([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    const kernel = [
      [-1, 1, -1],
      [1, 1, 1],
      [1, 1, 1],
    ];
    const clamp = getClamp(image);

    const interpolateBorder = getBorderInterpolation('reflect101', [0]);

    expect(
      computeConvolutionValue(1, 1, 0, image, kernel, interpolateBorder, {
        clamp,
      }),
    ).toBe(5);
  });

  it('return raw value', () => {
    const image = testUtils.createGreyImage([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    const kernel = [
      [0.5, 1, -1],
      [1, 1, 1],
      [1, 1, 1],
    ];
    const interpolateBorder = getBorderInterpolation('reflect101', [0]);

    expect(
      computeConvolutionValue(1, 1, 0, image, kernel, interpolateBorder, {
        returnRawValue: true,
      }),
    ).toBe(6.5);
  });
});

describe('rawDirectConvolution', () => {
  it('3x3 image and kernel', () => {
    const image = testUtils.createGreyImage([
      [1, 1, 20],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    const kernel = [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ];

    const expected = Float64Array.from([9, 28, 28, 9, 28, 28, 9, 9, 9]);

    expect(rawDirectConvolution(image, kernel)).toStrictEqual(expected);
  });

  it('3x3 image and kernel with floats', () => {
    const image = testUtils.createGreyImage([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    const kernel = [
      [1, 1, 1],
      [1, 0.5, 1],
      [1, 1, 1],
    ];

    const expected = Float64Array.from([
      8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,
    ]);

    expect(rawDirectConvolution(image, kernel)).toStrictEqual(expected);
  });
});
