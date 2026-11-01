import cv2
import numpy as np
from pathlib import Path



dir_path = Path('./images')
imgs_path = sorted(str(f) for f in dir_path.iterdir() if f.is_file())


print(imgs_path)


imgs = [cv2.imread(img) for img in imgs_path ]
# INTER_AREA averages each 10x10 block; the default INTER_LINEAR reads only 2x2 pixels 
original_imgs = [cv2.resize(img, dsize=None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA) for img in imgs]


#DEGRADATION: contrast reduction followed by gaussian noise

# we set beta to push back on the darkening of the image
a = 0.5
b = 128 * (1 - a)

noise_sigma = 5                   # std of the gaussian noise, in grey levels 
rng = np.random.default_rng(10)   # fixed seed

def degrade(img):
  reduced_contrast = cv2.convertScaleAbs(src=img, dst=None, alpha=a, beta=b)
  noisy = reduced_contrast.astype(np.float64) + rng.normal(0, noise_sigma, size=img.shape)
  return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


degraded_imgs = [degrade(img) for img in original_imgs]


#BASELINE 

out_dir = Path('./results/degraded_images')
out_dir.mkdir(parents=False, exist_ok=True)
for p, img in zip(imgs_path, degraded_imgs):
    out_path = out_dir / (Path(p).stem + "_degraded.png")   
    ok = cv2.imwrite(str(out_path), img)
    print(out_path, ok)


#LINEAR CONTRAST STRETCHING


def stretch_channel(c):
  f_min = np.min(c)
  f_max = np.max(c)

  if f_max == f_min:
    return c.copy()

  stretched = (c.astype(np.float32) - f_min) / (f_max - f_min) * 255.0
  return np.clip(stretched, 0, 255).astype(np.uint8)


def linear_contrast_stretching_per_channel(image):
  
  out = np.empty_like(image)
  for c in range(image.shape[2]):
    out[:,:,c] = stretch_channel(image[:,:,c])
  return out


def linear_contrast_stretching_luminance(image):
  lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  lab_img[:,:,0] = stretch_channel(lab_img[:,:,0])  
  return cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)


stretched_imgs = [linear_contrast_stretching_per_channel(img) for img in degraded_imgs]


out_dir = Path('./results/linear_contrast_stretching')
out_dir.mkdir(parents=False, exist_ok=True)
for p, img in zip(imgs_path, stretched_imgs):
    out_path = out_dir / (Path(p).stem + "_streched_channel.png")
    ok = cv2.imwrite(str(out_path), img)
    print(out_path, ok)


stretched_lum_imgs = [linear_contrast_stretching_luminance(img) for img in degraded_imgs]

out_dir = Path('./results/linear_contrast_stretching')
for p, img in zip(imgs_path, stretched_lum_imgs):
    out_path = out_dir / (Path(p).stem + "_streched_luminance.png")
    ok = cv2.imwrite(str(out_path), img)
    print(out_path, ok)


#HISTOGRAM EQUALIZATION (LAB)

def histogram_equalization_lab(image):
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  lab[:,:,0] = cv2.equalizeHist(lab[:,:,0]) 
  return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


hist_eq_lab_imgs = [histogram_equalization_lab(img) for img in degraded_imgs]


out_dir = Path('./results/histogram_equalization')
out_dir.mkdir(parents=False, exist_ok=True)
for p, img in zip(imgs_path, hist_eq_lab_imgs):
    out_path = out_dir / (Path(p).stem + "_hist_equal_lab.png")
    ok = cv2.imwrite(str(out_path), img)
    print(out_path, ok)


#Hist equal, seems to amplify Noise
#we try a denoise before hist equalization


#BILATERA FILTER + HE (LAB)

def bilateral_then_equalize(image, d=9, sigma_color=30, sigma_space=3):
  denoised = cv2.bilateralFilter(src=image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
  return histogram_equalization_lab(denoised)

combined_imgs = [bilateral_then_equalize(img) for img in degraded_imgs]

out_dir = Path('./results/bilateral_filter')
out_dir.mkdir(parents=False, exist_ok=True)
for p, img in zip(imgs_path, combined_imgs):
    out_path = out_dir / (Path(p).stem + "_denoise_and_equalize.png")
    ok = cv2.imwrite(str(out_path), img)
    print(out_path, ok)






#COMPARISON: PSNR and SSIM of every method against the original images
#the degraded row is the baseline, every method should be compared with it


import csv
from skimage.metrics import structural_similarity as ssim


methods = {
    'degraded (baseline)': degraded_imgs,
    'stretch per channel (BGR)': stretched_imgs,
    'stretch luminance (LAB)': stretched_lum_imgs,
    'hist eq (LAB)': hist_eq_lab_imgs,
    'bilateral + hist eq (LAB)': combined_imgs,
}

img_names = [Path(p).stem for p in imgs_path]

psnr_rows = []
ssim_rows = []
for name, enhanced_imgs in methods.items():
    psnr_rows.append([name] + [cv2.PSNR(original, enhanced) for original, enhanced in zip(original_imgs, enhanced_imgs)])
    ssim_rows.append([name] + [ssim(original, enhanced, channel_axis=2) for original, enhanced in zip(original_imgs, enhanced_imgs)])

#one row per (image, method)
out_path = Path('./results/metrics.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'method', 'psnr_db', 'ssim'])
    for psnr_row, ssim_row in zip(psnr_rows, ssim_rows):
        for j, name in enumerate(img_names):
            writer.writerow([name, psnr_row[0], f"{psnr_row[j + 1]:.2f}", f"{ssim_row[j + 1]:.3f}"])
print(out_path, 'written')


#SIDE BY SIDE COMPARISON: original | degraded | result

def add_label(img, text, header_h=40):
  header = np.full((header_h, img.shape[1], 3), 255, dtype=np.uint8)
  (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
  scale = min(0.8, (img.shape[1] - 20) / text_w)
  cv2.putText(header, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
  return np.vstack([header, img])

def side_by_side(original, degraded, result, method_name, gap_w=10):
  panels = [add_label(original, 'Original'), add_label(degraded, 'Degraded'), add_label(result, method_name)]
  gap = np.full((panels[0].shape[0], gap_w, 3), 255, dtype=np.uint8)
  return np.hstack([panels[0], gap, panels[1], gap, panels[2]])


comparison_methods = {
    'stretch_channel': ('Stretch per channel (BGR)', stretched_imgs),
    'stretch_luminance': ('Stretch luminance (LAB)', stretched_lum_imgs),
    'hist_eq_lab': ('Hist eq (LAB)', hist_eq_lab_imgs),
    'bilateral_hist_eq_lab': ('Bilateral + hist eq (LAB)', combined_imgs),
}

out_dir = Path('./results/comparisons')
out_dir.mkdir(parents=False, exist_ok=True)
for key, (label, result_imgs) in comparison_methods.items():
    for p, original, degraded, result in zip(imgs_path, original_imgs, degraded_imgs, result_imgs):
        out_path = out_dir / (Path(p).stem + "_" + key + ".png")
        ok = cv2.imwrite(str(out_path), side_by_side(original, degraded, result, label))
        print(out_path, ok)


#GRID: one figure per image with the original, the baseline and every method + the corresponding metrics

def add_label_metrics(img, title, metrics_text='', header_h=60):
  header = np.full((header_h, img.shape[1], 3), 255, dtype=np.uint8)
  for text, y, max_scale, thickness in [(title, 25, 0.7, 2), (metrics_text, 50, 0.55, 1)]:
    if not text:
      continue
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness)
    scale = min(max_scale, (img.shape[1] - 20) / text_w)
    cv2.putText(header, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
  return np.vstack([header, img])

def make_grid(panels, n_cols=3, gap_w=10):
  h, w = panels[0].shape[:2]
  n_rows = (len(panels) + n_cols - 1) // n_cols
  grid = np.full((n_rows * h + (n_rows - 1) * gap_w, n_cols * w + (n_cols - 1) * gap_w, 3), 255, dtype=np.uint8)
  for i, panel in enumerate(panels):
    r, c = divmod(i, n_cols)
    grid[r * (h + gap_w): r * (h + gap_w) + h, c * (w + gap_w): c * (w + gap_w) + w] = panel
  return grid



out_dir = Path('./results/grids')
out_dir.mkdir(parents=False, exist_ok=True)
for j, (p, original) in enumerate(zip(imgs_path, original_imgs)):
    panels = [add_label_metrics(original, 'Original (reference)')]
    for i, (name, method_imgs) in enumerate(methods.items()):
        metrics_text = f"PSNR {psnr_rows[i][j + 1]:.2f} dB | SSIM {ssim_rows[i][j + 1]:.3f}"
        panels.append(add_label_metrics(method_imgs[j], name, metrics_text))
    out_path = out_dir / (Path(p).stem + "_grid.png")
    ok = cv2.imwrite(str(out_path), make_grid(panels))
    print(out_path, ok)
