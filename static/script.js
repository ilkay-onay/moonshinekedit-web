const cameraFeed = document.getElementById('camera_feed');
const processedFeed = document.getElementById('processed_feed');
const originalImage = document.getElementById('original_image');
const generatedImage = document.getElementById('generated_image');
const canvasCtx = processedFeed.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        cameraFeed.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing camera', err);
    });

function captureFrame() {
    canvasCtx.drawImage(cameraFeed, 0, 0);
    const imgData = canvasCtx.getImageData(0, 0, 640, 480);
    const src = cv.matFromImageData(imgData);
    const dst = new cv.Mat();

    if (document.getElementById('thr_otsu').checked) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        cv.threshold(src, dst, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    } else if (document.getElementById('thr_value').value && document.getElementById('thr_block_size').value && document.getElementById('thr_c').value) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        cv.adaptiveThreshold(src, dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, parseInt(document.getElementById('thr_block_size').value), parseFloat(document.getElementById('thr_c').value));
    } else if (document.getElementById('thr_value').value) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        cv.threshold(src, dst, parseFloat(document.getElementById('thr_value').value), 255, cv.THRESH_BINARY);
    }

    if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_filter').checked) {
        cv.boxFilter(src, dst, -1, new cv.Size(parseInt(document.getElementById('bl_kernelsize').value), parseInt(document.getElementById('bl_kernelsize').value)));
    } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_gaussian').checked) {
        cv.GaussianBlur(src, dst, new cv.Size(parseInt(document.getElementById('bl_kernelsize').value), parseInt(document.getElementById('bl_kernelsize').value)), 0);
    } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_normal').checked) {
        cv.blur(src, dst, new cv.Size(parseInt(document.getElementById('bl_kernelsize').value), parseInt(document.getElementById('bl_kernelsize').value)));
    } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_bilateral').checked) {
        cv.bilateralFilter(src, dst, parseInt(document.getElementById('bl_kernelsize').value), 75, 75);
    } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_median').checked) {
        cv.medianBlur(src, dst, parseInt(document.getElementById('bl_kernelsize').value));
    }

    if (document.getElementById('gmm_value').value) {
        const gamma = parseFloat(document.getElementById('gmm_value').value);
        const invGamma = 1.0 / gamma;
        const lut = new cv.Mat(256, 1, cv.CV_8UC1);
        for (let i = 0; i < 256; i++) {
            lut.ucharPtr(i)[0] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
        }
        cv.LUT(src, lut, dst);
    }

    if (document.getElementById('bda_sobel').checked) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        const gradX = new cv.Mat();
        const gradY = new cv.Mat();
        cv.Sobel(src, gradX, cv.CV_64F, 1, 0, 5);
        cv.Sobel(src, gradY, cv.CV_64F, 0, 1, 5);
        cv.magnitude(gradX, gradY, dst);
        cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    } else if (document.getElementById('bda_laplacian').checked) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        cv.Laplacian(src, dst, cv.CV_64F, 1, 1, 0);
        cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    } else if (document.getElementById('bda_canny').checked) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        cv.Canny(src, dst, 100, 200);
    } else if (document.getElementById('bda_deriche').checked) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        const alpha = 0.5;
        const kernelSize = 3;
        const kx = cv.getDerivKernels(1, 1, kernelSize, true);
        const ky = cv.getDerivKernels(1, 1, kernelSize, true);
        const dericheKernelX = kx.mul(new cv.Mat().ones(kx.rows, kx.cols, kx.type()), alpha);
        const dericheKernelY = ky.mul(new cv.Mat().ones(ky.rows, ky.cols, ky.type()), alpha);
        const dericheX = new cv.Mat();
        const dericheY = new cv.Mat();
        cv.filter2D(src, dericheX, -1, dericheKernelX);
        cv.filter2D(src, dericheY, -1, dericheKernelY);
        cv.magnitude(dericheX, dericheY, dst);
        cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
    } else if (document.getElementById('bda_harris').checked) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        const gray = new cv.Mat();
        src.convertTo(gray, cv.CV_32F);
        const corners = new cv.Mat();
        cv.cornerHarris(gray, corners, 2, 3, 0.04);
        cv.dilate(corners, corners, new cv.Mat());
        const minMax = cv.minMaxLoc(corners);
        const maxVal = minMax.maxVal;
        for (let i = 0; i < corners.rows; i++) {
            for (let j = 0; j < corners.cols; j++) {
                if (corners.ucharPtr(i, j)[0] > 0.01 * maxVal) {
                    cv.circle(src, new cv.Point(j, i), 5, [0, 0, 255, 255], 2);
                }
            }
        }
        src.copyTo(dst);
    }

    if (document.getElementById('dt_viola').checked) {
    } else if (document.getElementById('dt_kontur').checked) {
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
        const edges = new cv.Mat();
        cv.Canny(src, edges, 50, 150);
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        cv.drawContours(src, contours, -1, [0, 255, 0, 255], 2);
        src.copyTo(dst);
    } else if (document.getElementById('dt_watershed').checked) {
    }

    cv.imshow(processedFeed, dst);
    src.delete();
    dst.delete();
}

setInterval(captureFrame, 1000/15);

document.getElementById('btn_loadimg').addEventListener('click', function() {
    const input = document.createElement('input');
    input.type = 'file';
    input.onchange = function(event) {
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
            originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    };
    input.click();
});

document.getElementById('btn_savepar').addEventListener('click', function() {
    const parameters = {
        thr_value: document.getElementById('thr_value').value,
        thr_block_size: document.getElementById('thr_block_size').value,
        thr_c: document.getElementById('thr_c').value,
        thr_otsu: document.getElementById('thr_otsu').checked,
        bl_kernelsize: document.getElementById('bl_kernelsize').value,
        bl_filter: document.getElementById('bl_filter').checked,
        bl_gaussian: document.getElementById('bl_gaussian').checked,
        bl_normal: document.getElementById('bl_normal').checked,
        bl_bilateral: document.getElementById('bl_bilateral').checked,
        bl_median: document.getElementById('bl_median').checked,
        gmm_value: document.getElementById('gmm_value').value,
        bda_sobel: document.getElementById('bda_sobel').checked,
        bda_laplacian: document.getElementById('bda_laplacian').checked,
        bda_canny: document.getElementById('bda_canny').checked,
        bda_deriche: document.getElementById('bda_deriche').checked,
        bda_harris: document.getElementById('bda_harris').checked,
        dt_viola: document.getElementById('dt_viola').checked,
        dt_kontur: document.getElementById('dt_kontur').checked,
        dt_watershed: document.getElementById('dt_watershed').checked
    };
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(parameters));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "parameters.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
});

document.getElementById('btn_loadpar').addEventListener('click', function() {
    const input = document.createElement('input');
    input.type = 'file';
    input.onchange = function(event) {
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
            const parameters = JSON.parse(e.target.result);
            document.getElementById('thr_value').value = parameters.thr_value;
            document.getElementById('thr_block_size').value = parameters.thr_block_size;
            document.getElementById('thr_c').value = parameters.thr_c;
            document.getElementById('thr_otsu').checked = parameters.thr_otsu;
            document.getElementById('bl_kernelsize').value = parameters.bl_kernelsize;
            document.getElementById('bl_filter').checked = parameters.bl_filter;
            document.getElementById('bl_gaussian').checked = parameters.bl_gaussian;
            document.getElementById('bl_normal').checked = parameters.bl_normal;
            document.getElementById('bl_bilateral').checked = parameters.bl_bilateral;
            document.getElementById('bl_median').checked = parameters.bl_median;
            document.getElementById('gmm_value').value = parameters.gmm_value;
            document.getElementById('bda_sobel').checked = parameters.bda_sobel;
            document.getElementById('bda_laplacian').checked = parameters.bda_laplacian;
            document.getElementById('bda_canny').checked = parameters.bda_canny;
            document.getElementById('bda_deriche').checked = parameters.bda_deriche;
            document.getElementById('bda_harris').checked = parameters.bda_harris;
            document.getElementById('dt_viola').checked = parameters.dt_viola;
            document.getElementById('dt_kontur').checked = parameters.dt_kontur;
            document.getElementById('dt_watershed').checked = parameters.dt_watershed;
        };
        reader.readAsText(file);
    };
    input.click();
});
document.getElementById('btn_generateimg').addEventListener('click', function() {
    const originalImageSrc = originalImage.src;
    if (originalImageSrc) {
        const img = new Image();
        img.src = originalImageSrc;
        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const imgData = ctx.getImageData(0, 0, img.width, img.height);
            const src = cv.matFromImageData(imgData);
            const dst = new cv.Mat();

            if (document.getElementById('thr_otsu').checked) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                cv.threshold(src, dst, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
            } else if (document.getElementById('thr_value').value && document.getElementById('thr_block_size').value && document.getElementById('thr_c').value) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                cv.adaptiveThreshold(src, dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, parseInt(document.getElementById('thr_block_size').value), parseFloat(document.getElementById('thr_c').value));
            } else if (document.getElementById('thr_value').value) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                cv.threshold(src, dst, parseFloat(document.getElementById('thr_value').value), 255, cv.THRESH_BINARY);
            }

            if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_filter').checked) {
                cv.boxFilter(src, dst, -1, new cv.Size(parseInt(document.getElementById('bl_kernelsize').value), parseInt(document.getElementById('bl_kernelsize').value)));
            } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_gaussian').checked) {
                cv.GaussianBlur(src, dst, new cv.Size(parseInt(document.getElementById('bl_kernelsize').value), parseInt(document.getElementById('bl_kernelsize').value)), 0);
            } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_normal').checked) {
                cv.blur(src, dst, new cv.Size(parseInt(document.getElementById('bl_kernelsize').value), parseInt(document.getElementById('bl_kernelsize').value)));
            } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_bilateral').checked) {
                cv.bilateralFilter(src, dst, parseInt(document.getElementById('bl_kernelsize').value), 75, 75);
            } else if (document.getElementById('bl_kernelsize').value && document.getElementById('bl_median').checked) {
                cv.medianBlur(src, dst, parseInt(document.getElementById('bl_kernelsize').value));
            }

            if (document.getElementById('gmm_value').value) {
                const gamma = parseFloat(document.getElementById('gmm_value').value);
                const invGamma = 1.0 / gamma;
                const lut = new cv.Mat(256, 1, cv.CV_8UC1);
                for (let i = 0; i < 256; i++) {
                    lut.ucharPtr(i)[0] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
                }
                cv.LUT(src, lut, dst);
            }

            if (document.getElementById('bda_sobel').checked) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                const gradX = new cv.Mat();
                const gradY = new cv.Mat();
                cv.Sobel(src, gradX, cv.CV_64F, 1, 0, 5);
                cv.Sobel(src, gradY, cv.CV_64F, 0, 1, 5);
                cv.magnitude(gradX, gradY, dst);
                cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
            } else if (document.getElementById('bda_laplacian').checked) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                cv.Laplacian(src, dst, cv.CV_64F, 1, 1, 0);
                cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
            } else if (document.getElementById('bda_canny').checked) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                cv.Canny(src, dst, 100, 200);
            } else if (document.getElementById('bda_deriche').checked) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                const alpha = 0.5;
                const kernelSize = 3;
                const kx = cv.getDerivKernels(1, 1, kernelSize, true);
                const ky = cv.getDerivKernels(1, 1, kernelSize, true);
                const dericheKernelX = kx.mul(new cv.Mat().ones(kx.rows, kx.cols, kx.type()), alpha);
                const dericheKernelY = ky.mul(new cv.Mat().ones(ky.rows, ky.cols, ky.type()), alpha);
                const dericheX = new cv.Mat();
                const dericheY = new cv.Mat();
                cv.filter2D(src, dericheX, -1, dericheKernelX);
                cv.filter2D(src, dericheY, -1, dericheKernelY);
                cv.magnitude(dericheX, dericheY, dst);
                cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
            } else if (document.getElementById('bda_harris').checked) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                const gray = new cv.Mat();
                src.convertTo(gray, cv.CV_32F);
                const corners = new cv.Mat();
                cv.cornerHarris(gray, corners, 2, 3, 0.04);
                cv.dilate(corners, corners, new cv.Mat());
                const minMax = cv.minMaxLoc(corners);
                const maxVal = minMax.maxVal;
                for (let i = 0; i < corners.rows; i++) {
                    for (let j = 0; j < corners.cols; j++) {
                        if (corners.ucharPtr(i, j)[0] > 0.01 * maxVal) {
                            cv.circle(src, new cv.Point(j, i), 5, [0, 0, 255, 255], 2);
                        }
                    }
                }
                src.copyTo(dst);
            }

            if (document.getElementById('dt_viola').checked) {
            } else if (document.getElementById('dt_kontur').checked) {
                cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
                const edges = new cv.Mat();
                cv.Canny(src, edges, 50, 150);
                const contours = new cv.MatVector();
                const hierarchy = new cv.Mat();
                cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
                cv.drawContours(src, contours, -1, [0, 255, 0, 255], 2);
                src.copyTo(dst);
            } else if (document.getElementById('dt_watershed').checked) {
            }

            cv.imshow(canvas, dst);
            generatedImage.src = canvas.toDataURL();
            src.delete();
            dst.delete();
        };
    }
});

document.getElementById('btn_saveimg').addEventListener('click', function() {
    const generatedImageSrc = generatedImage.src;
    if (generatedImageSrc) {
        const link = document.createElement('a');
        link.href = generatedImageSrc;
        link.download = 'generated_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});

document.getElementById('btn_reset').addEventListener('click', function() {
    document.getElementById('thr_value').value = '';
    document.getElementById('thr_block_size').value = '';
    document.getElementById('thr_c').value = '';
    document.getElementById('thr_otsu').checked = false;
    document.getElementById('bl_kernelsize').value = '';
    document.getElementById('bl_filter').checked = false;
    document.getElementById('bl_gaussian').checked = false;
    document.getElementById('bl_normal').checked = false;
    document.getElementById('bl_bilateral').checked = false;
    document.getElementById('bl_median').checked = false;
    document.getElementById('gmm_value').value = '';
    document.getElementById('bda_sobel').checked = false;
    document.getElementById('bda_laplacian').checked = false;
    document.getElementById('bda_canny').checked = false;
    document.getElementById('bda_deriche').checked = false;
    document.getElementById('bda_harris').checked = false;
    document.getElementById('dt_viola').checked = false;
    document.getElementById('dt_kontur').checked = false;
    document.getElementById('dt_watershed').checked = false;
});

document.getElementById('btn_exit').addEventListener('click', function() {
    fetch('/exit', {
        method: 'POST'
    }).then(response => response.json())
      .then(data => {
          if (data.success) {
              alert('Exiting the application');
              window.close();
          }
      });
});