# Introduction

**NOTE**  
the original images serve as references 
for evaluating the synthetic degradations, a clean reference may not be available in
real-world scenarios.

# OBJECTIVE 
In this HomeWork we will implement a small pipeline for enhancing degarded images isomg classical image processing methods.
The objective is to analyze and evaluate the effects of the methods on the degraded images, understand and explain the reasons for such effects and justify the choices that are made.

the 4 methods we will implement are the following:

- **baseline**: the degraded images, wich are a result of a contrast reduction followed by a small amount of gaussian noise on the original images.

- **linear contrast stretching**: operating on both the BGR channels and on Luminance in the LAB format.

- **histogram_equalization**: equalizing L on LAB.

- **bilateral filter + histogram equalization (LAB)**: the idea is to reduce the noise amplification of the histogram equalization by first applying a classical denoiser, this resulted in a reduction in noise, but also noticeable smoothing of the details compared to the single histogram equalization on L.


# IMAGES AND SOURCES 
 
each image has been chosen based on a particular condition:

All images under unsplash license:
https://unsplash.com/license


low_contrast: https://unsplash.com/photos/a-blurry-photo-of-a-person-holding-a-cell-phone-RxQfhv2sBlU
shadows: https://unsplash.com/photos/a-wall-with-a-painting-on-it-cuiNuOWzilU
poor_details: https://unsplash.com/photos/nighttime-cityscape-with-glowing-lanterns-and-lights-ZrDltpoqExE


# METHODS AND VISUAL COMPARISON

Each figure shows, from left to right:
- the original image (resized to 10%)
- the degraded image (contrast reduction with α = 0.5, β = 64, followed by Gaussian noise with σ = 5)
- the result of the corresponding method.

## Preprocessing and degradation parameters

| Step | Code | Parameters |
|---|---|---|
| Resize | `cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)` | factor 0.1, `INTER_AREA` |
| Contrast reduction | `cv2.convertScaleAbs(img, alpha=0.5, beta=64)` | α = 0.5, β = 128·(1 − α) = 64 |
| Gaussian noise | `rng.normal(0, 5, size=img.shape)` with `rng = np.random.default_rng(10)` | σ = 5 grey levels, seed 10 |

`convertScaleAbs` returns an 8-bit image; the noise is added in floating point, and the result is rounded and clipped to [0, 255] again.

**Resize to 10%.** At 10% scale every method runs in a fraction of a second, and the shadows, folds and lights are still clearly visible. All window sizes below (bilateral filter, SSIM) are chose with respect to the resized image.

**Interpolation: `INTER_AREA` instead of the default `INTER_LINEAR`.** `INTER_LINEAR` computes each output pixel from the 2×2 nearest input pixels. The OpenCV documentation recommends `INTER_AREA` for shrinking, because it averages the whole 10×10 block. Noise measured in the reference images¹:


**Contrast reduction.**
- α = 0.5 halves every intensity difference. The contrast loss is strong, but every edge is still present, so the methods have something to recover.
- β = 128·(1 − α) = 64, Every pixel moves halfway toward 128 and the image is not darkened. 
This explains why the baseline PSNR depends so much on the image. low_contrast is already close to mid-grey so the degradation changes it little. 
Shadows and poor_details span almost 0–255 so the difference after the degradation is more noticeable. 

- `cv2.convertScaleAbs` computes |α·I + β| on B, G and R independently, then rounds and saturates to [0, 255]. 

**Gaussian noise.** 
- σ = 5 grey levels, drawn independently for every pixel and channel. 
- The noise is added after the contrast reduction, as a sensor adds noise to an already dim, flat scene.
- **Seed.** The noise comes from `np.random.default_rng(10)`, so every run produces the same degraded images. 


# Linear contrast stretching
we apply Linear contrast stretching, both Per-Channel for BGR and on Luminance for LAB
following the formula:

stretched = (image - f_min) / (f_max - f_min) * (new_max - new_min) + new_min

## Parameters

f_min and f_max are the highest and lowes values in the degraded image, computed separately for B, G and R (per-channel variant) or on L only (luminance variant). new_min = 0 and new_max = 255 use the whole 8-bit range.


- **Noise and color, per channel.** Each channel gets its own gain, so the balance between B, G and R changes. In low_contrast, G gets the lowest gain so green is weakened relative to blue and red, which produces the blue/purple cast. In Shadows, B is stretched more than R, so the warm wall turns slightly cooler. 
The noise of every channel is amplified too.
- **Noise and color, luminance.** Only L is amplified, so the noise in a and b stays at its degraded level. 
Stretching L does not restore the degradation in color, so the colors look faded.
    

## Per channel (BGR)

**Shadows**

![Shadows: original, degraded, stretch per channel](./results/comparisons/Shadows_stretch_channel.png)

**poor_details**

![poor_details: original, degraded, stretch per channel](./results/comparisons/poor_details_stretch_channel.png)

**low_contrast**

![low_contrast: original, degraded, stretch per channel](./results/comparisons/low_contrast_stretch_channel.png)


## Luminance (L channel in LAB)

**Shadows**

![Shadows: original, degraded, stretch luminance](./results/comparisons/Shadows_stretch_luminance.png)

**poor_details**

![poor_details: original, degraded, stretch luminance](./results/comparisons/poor_details_stretch_luminance.png)

**low_contrast**

![low_contrast: original, degraded, stretch luminance](./results/comparisons/low_contrast_stretch_luminance.png)




# Histogram equalization on L channel (LAB)

**Shadows**

![Shadows: original, degraded, histogram equalization LAB](./results/comparisons/Shadows_hist_eq_lab.png)

**poor_details**

![poor_details: original, degraded, histogram equalization LAB](./results/comparisons/poor_details_hist_eq_lab.png)

**low_contrast**

![low_contrast: original, degraded, histogram equalization LAB](./results/comparisons/low_contrast_hist_eq_lab.png)


# Bilateral filter + histogram equalization (LAB)

parameters for bilateral filter:
- d=9,  diameter of each pixel neighborhood.
- sigma_color=30,  standard deviation in the color space.
- sigma_space=3, standard deviation in the coordinate space.

reasoning:

HE amplifies small differences, so a denoiser is applied first to reduce what HE will amplify. The bilateral filter replaces each pixel with a weighted mean of its neighbours. Each weight is the product of a spatial Gaussian (distance, sigma_space) and a range Gaussian (color difference, sigma_color), so flat regions are smoothed but strong edges are not.

- **d = 9.** The OpenCV documentation recommends d = 5 for real-time use and d = 9 for offline applications that need heavy noise filtering. Our images are few and small, so the cost is negligible.
- **sigma_color = 30.** The OpenCV documentation places useful values between 10 (little effect) and 150 (cartoon-like).
- **sigma_space = 3.** When d > 0, OpenCV takes the window size from d, not from sigma_space. so this parameter has no effect.


**Shadows**

![Shadows: original, degraded, bilateral filter + histogram equalization LAB](./results/comparisons/Shadows_bilateral_hist_eq_lab.png)

**poor_details**

![poor_details: original, degraded, bilateral filter + histogram equalization LAB](./results/comparisons/poor_details_bilateral_hist_eq_lab.png)

**low_contrast**

![low_contrast: original, degraded, bilateral filter + histogram equalization LAB](./results/comparisons/low_contrast_bilateral_hist_eq_lab.png)






# METRICS
all the metrics are a result of the comparison of the method with the corresponding original image

**Metric parameters.**
- **PSNR**: `cv2.PSNR(original, result)` with the default peak R = 255. The MSE is averaged over all three channels.
- **SSIM**: `skimage.metrics.structural_similarity(original, result, channel_axis=2)` library default: 7×7 uniform window, K1 = 0.01, K2 = 0.03, data range 255 (inferred from uint8). SSIM is computed on B, G and R separately and then averaged.

**PSNR (dB)**
| method                    |   Shadows |   low_contrast |   poor_details |
|---------------------------|-----------|----------------|----------------|
| degraded (baseline)       |     16.25 |          26.82 |          14.17 |
| stretch per channel (BGR) |     21.47 |          16.83 |          25.80 |
| stretch luminance (LAB)   |     22.58 |          20.01 |          26.83 |
| hist eq (LAB)             |     17.66 |          11.95 |           7.70 |
| bilateral + hist eq (LAB) |     17.77 |          12.00 |           6.37 |

**SSIM**
| method                    |   Shadows |   low_contrast |   poor_details |
|---------------------------|-----------|----------------|----------------|
| degraded (baseline)       |     0.654 |          0.706 |          0.487 |
| stretch per channel (BGR) |     0.642 |          0.213 |          0.538 |
| stretch luminance (LAB)   |     0.714 |          0.299 |          0.674 |
| hist eq (LAB)             |     0.513 |          0.072 |          0.076 |
| bilateral + hist eq (LAB) |     0.730 |          0.401 |          0.071 |


# GRID COMPARISON


![low_contrast](./results/grids/low_contrast_grid.png)

![poor_details](./results/grids/poor_details_grid.png)

![shadows](./results/grids/Shadows_grid.png)



# Numerical improvement vs. visual quality

PSNR and SSIM are full-reference metrics: they measure how close a result is to the original, not how good it looks.
A higher score therefore does not always mean a better-looking image.

## Shadows, bilateral filter + histogram equalization (LAB)

This method has the highest SSIM of all on Shadows (0.730, against 0.714 for luminance stretching and 0.654 for the baseline), and its PSNR is above the baseline (17.77 against 16.25 dB).
But visually it is worse than luminance stretching:
- the texture of the lower wall is flattened into low-detail patches.
- the light patches lose their warm color.
- the change in intensity between shadows and light is too drastic.

why SSIM rewards it:
SSIM compares the two images in small windows and, in each window, **multiplies** three terms:
- **luminance**: local mean brightness similarity
- **contrast**: local amount of variation (standard deviation) similarity
- **structure**: correlation in pixel values.

In a flat window of the original (like a uniform piece of wall), any noise in the result adds variation that the original does not have, so the contrast and structure terms drop. The bilateral filter removes that noise, so these windows score high again. 

The bilateral filter wins only in the flat windows, where it removes noise, and loses in the textured ones, where it removes detail. 
SSIM averages the two, so the gain on flat areas outweighs the loss of the texture we can actually see.

## Shadows, histogram equalization (LAB): the two metrics disagree

PSNR improves over the baseline (16.25 → 17.66 dB), but SSIM gets worse (0.654 → 0.513). Visually the image presents noticeable issues:
- the grain on the upper wall is heavily amplified.
- the light patches lose their warm color and turn almost white.
- once again, the change in intensity between shadows and light is too drastic.

why PSNR improves:  
the degradation pulls every pixel halfway toward 128, which halves the spread in intensity, so the degraded image is wrong almost everywhere.
Equalization spreads the values out again, so most pixels move closer to their original value and the mean squared error drops.

why SSIM gets worse: equalization repairs the luminance and contrast terms, the two the degradation had damaged, but it amplifies the noise, so the structure term collapses. ssim score is the average of the three components, the collapse in the structure term, outweighs the gain in luminance and contrast.

## low_contrast: the opposite case

On low_contrast, the baseline has the best PSNR and SSIM (26.82 dB / 0.706), yet it is the version where the folds are hardest to see. Every method lowers both metrics, because the original itself has low contrast: any enhancement moves the image away from the reference, even when it makes it easier to read.


# Qualitative analysis

## Shadows: linear stretching on luminance (LAB)

- **Easier to see:** the edges of the shadows, the outline of the pole, the silhouette of the person and the texture of the wall. The dark wall keeps its neutral tone. This method has the best PSNR (22.58 dB), the best SSIM after bilateral + HE, and the smallest  difference in color form the original, second only to linear stretchin per channel.
- **Trade-offs:** the noise is amplified and is visible as fine grain on the wall. The colours are slightly less saturated than in the original, because stretching only L does not restore the colors.
- **Other methods:** 
  - *per-channel stretching* gives similar contrast, but it also amplifies the noise.
  - *Histogram equalization* amplifies the grain on the wall and turns the light patches white. Applying the bilateral filter first removes the grain, but flattens the texture of the wall and does not fix the white lights.

## low_contrast: linear stretching on luminance (LAB)

- **Easier to see:** the folds and creases of the sheet, which are barely visible in the degraded image.
- **Trade-offs:**  the noise is amplified as much as the folds so the image is grainy. The colours look more grey than in the original. Like before, stretching only L does not restore the colors.
- **Other methods:** 
  - *per-channel stretching* produces unnatural colours, a strong blue and purple "filter", plus a colored grain. 
  - *Histogram equalization* shows the creases with the most contrast, but it amplifies the grain even more.
  - *bilateral filter + HE*gives the cleanest image (without the faded look and noise), the large folds are easiest to see, but details like small creases are lost, the borders also become dark.

## poor_details: linear stretching per channel (BGR)

- **Easier to see:** the lights in the image regain their contrast against the dark background, with colours closer to the original.
- **Trade-offs:** faint details on the buildings are lost. the image presents a visible grain that contrasts heavily with the dark background.
- **Other methods:**
  - *Luminance stretching (LAB)* gives almost the same image with less colored grain, and it has the highest PSNR and SSIM (26.83 dB / 0.674), but the colours look faded: The degradation pulled the colours toward grey, and stretching only L restores the brightness but not the colour.
  - *Histogram equalization (LAB)* fails on this image. Equalization tries to spread the pixels evenly over all Luminance levels. Here most of the image is dark, which in the degraded image results in a few L levels that differ only by the noise. Equalization spreads them over half of the output range, so the background turns into salt-and-pepper noise. It highlights some details on the left, but at the cost of the rest of the image.
  - *Bilateral filter + histogram equalization (LAB)*: also fails. The filter reduces the noise, but the steep equalization mapping still amplifies the residual variations into large blotches: the problem is not only the noise, but the way equalization redistributes a histogram concentrated on a few values.

² Mean color difference ΔE (CIE76, Euclidean distance in Lab) between result and original, measured after smoothing both with a Gaussian (σ = 3 px) so that the noise is not counted.



