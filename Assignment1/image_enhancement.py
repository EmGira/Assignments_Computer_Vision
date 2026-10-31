import cv2
import numpy as np
from pathlib import Path



dir_path = Path('./images')
imgs_path = [ str(f) for f in dir_path.iterdir() if f.is_file() ]


print(imgs_path)


imgs = [cv2.imread(img) for img in imgs_path ]
original_imgs = [cv2.resize(img, dsize=None, fx=0.1, fy=0.1) for img in imgs]



cv2.waitKey(0)
cv2.destroyAllWindows()


blurred_imgs = [cv2.GaussianBlur(img, (5,5), 0) for img in original_imgs]


cv2.waitKey(0)
cv2.destroyAllWindows()


a = 0.5

# we set beta to push back on the darkening of the image
b = 128 * (1 - a)

reduces_contrast_imgs = [cv2.convertScaleAbs(src=img, dst=None, alpha=a, beta=b) for img in blurred_imgs]



degraded_imgs = reduces_contrast_imgs

cv2.waitKey(0)
cv2.destroyAllWindows()


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

# linear stretching fails on poor_details.png: its range is already almost full (bright lights), so it only undoes the degradation; the histogram stays crowded in the dark end


#Histogram equalization


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


#Bilateral filter + histogram equalization (LAB)
#we try a denoise before hist equalization

def bilateral_then_equalize(image, d=9, sigma_color=30, sigma_space=5):
  denoised = cv2.bilateralFilter(src=image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
  return histogram_equalization_lab(denoised)

combined_imgs = [bilateral_then_equalize(img) for img in degraded_imgs]

out_dir = Path('./results/bilateral_filter')
out_dir.mkdir(parents=False, exist_ok=True)
for p, img in zip(imgs_path, combined_imgs):
    out_path = out_dir / (Path(p).stem + "_denoise_and_equalize.png")
    ok = cv2.imwrite(str(out_path), img)
    print(out_path, ok)


#PSNR original-degraded
#SSIM
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate

imgs_psnr = [ cv2.PSNR(original, degraded) for original, degraded in zip(original_imgs, degraded_imgs)]



#COMPARISON: PSNR and SSIM of every method against the original images
#the degraded row is the baseline, every method should be compared with it

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

print("PSNR (dB)")
print(tabulate(psnr_rows, headers=['method'] + img_names, floatfmt='.2f', tablefmt='github'))
print()
print("SSIM")
print(tabulate(ssim_rows, headers=['method'] + img_names, floatfmt='.3f', tablefmt='github'))


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
