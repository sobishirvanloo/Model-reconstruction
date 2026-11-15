# MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
# See the LICENSE file in the project root for the terms of use.
"""Persian user interface (View ▸ Language).

The program's texts are written in English.  With Persian selected,
``install`` makes every text the user sees go through ``tr``:

* the setters the program calls (``setText``, ``setToolTip``, menu and
  combo items, table headers, message boxes …) are wrapped once, so texts
  set at any time are translated;
* texts that widgets got in their constructor or from the ``.ui`` file
  are translated when their window is first shown (``_Translator``);
* the layout runs right to left.

``tr`` looks a text up in ``FA``; section headers (upper case, with the
▾ / ▸ fold marker), menu mnemonics (&) and shortcuts after a tab are
handled, and ``PATTERNS`` translate texts that contain numbers or names.
Unknown texts stay English; the log (console) and technical names
(algorithms, methods, RMSE …) stay English on purpose.

Switching the language restarts the program (``settings.language``).
"""

import re

from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, QObject, Qt

ENGLISH, PERSIAN = "en", "fa"
LANGUAGES = {ENGLISH: "English", PERSIAN: "فارسی"}

FA = {
    # ---------- tabs, sections ----------
    "1   Import mesh": "۱   وارد کردن مش",
    "2   Process points": "۲   پردازش نقاط",
    "3   Reconstruct": "۳   بازسازی",
    "4   Compare": "۴   مقایسه",
    "FILE": "فایل",
    "ROTATION": "چرخش",
    "POSITION": "مکان",
    "MODELS": "مدل‌ها",
    "COMPARISON METHODS": "روش‌های مقایسه",
    "DISPLAY": "نمایش",
    "COLOURS": "رنگ‌ها",
    "OUTPUT": "خروجی",
    "RECONSTRUCTION": "بازسازی",
    "PREPROCESSING": "پیش‌پردازش",
    "POST-PROCESSING": "پس‌پردازش",
    "DENSITY & SAMPLING": "چگالی و نمونه‌برداری",
    "Errors": "خطاها",
    "Distribution": "توزیع",
    "Object info": "اطلاعات شیء",
    # ---------- common buttons / labels ----------
    "Copy": "کپی",
    "Open file…": "باز کردن فایل…",
    "Open a file…": "باز کردن فایل…",
    "Open point cloud…": "باز کردن ابر نقاط…",
    "or open the sample bunny": "یا باز کردن نمونه‌ی خرگوش",
    "No file loaded": "هیچ فایلی باز نیست",
    "No data loaded": "داده‌ای بارگذاری نشده",
    "No model loaded": "مدلی بارگذاری نشده",
    "No reference loaded": "مرجعی بارگذاری نشده",
    "No file": "بدون فایل",
    "Remove data": "حذف داده",
    "Export points…": "ذخیره‌ی نقاط…",
    "Export mesh…": "ذخیره‌ی مش…",
    "Send to Processing": "ارسال به پردازش",
    "Send to Reconstruction": "ارسال به بازسازی",
    "Send to Compare": "ارسال به مقایسه",
    "Quality report": "گزارش کیفیت",
    "Shuffle point order": "بُر زدن ترتیب نقاط",
    "Degrade data…": "تخریب داده…",
    "Voxel size (0 = original density)": "اندازه‌ی وکسل (۰ = چگالی اصلی)",
    "Algorithm": "الگوریتم",
    "Optimize parameters…": "بهینه‌سازی پارامترها…",
    "Stop reconstruction": "توقف بازسازی",
    "Depth": "عمق",
    "Width": "عرض",
    "Scale": "مقیاس",
    "Linear_fit": "برازش خطی",
    "Preprocess": "پیش‌پردازش",
    "Denoise + orient normals": "حذف نویز + جهت‌دهی نرمال‌ها",
    "Voxel size (0 = off)": "اندازه‌ی وکسل (۰ = خاموش)",
    "Density filter (%)": "فیلتر چگالی (٪)",
    "Crop to bounds": "برش به محدوده‌ی داده",
    "Smoothing (iters)": "هموارسازی (تکرار)",
    "Rotation °": "چرخش °",
    "Position": "مکان",
    "X-Axis (angle)": "محور X (زاویه)",
    "Y-Axis (angle)": "محور Y (زاویه)",
    "Z-Axis (angle)": "محور Z (زاویه)",
    "X-Axis (position)": "محور X (مکان)",
    "Y-Axis (position)": "محور Y (مکان)",
    "Z-Axis (position)": "محور Z (مکان)",
    "Reset rotation": "بازنشانی چرخش",
    "Reset position": "بازنشانی مکان",
    # ---------- viewport ----------
    "Reset view": "بازنشانی نما",
    "View": "نما",
    "Object": "شیء",
    "Background": "پس‌زمینه",
    "Pose": "ژست",
    "Screenshot": "تصویر صفحه",
    "Drag: rotate": "کشیدن: چرخش",
    "Drag: move": "کشیدن: جابه‌جایی",
    "Scroll: zoom": "چرخ موس: بزرگ‌نمایی",
    "Front": "جلو",
    "Top": "بالا",
    "Side": "کنار",
    "Iso": "ایزومتریک",
    "Colour & display…": "رنگ و نمایش…",
    "Colour & guides…": "رنگ و راهنماها…",
    "Restore default": "بازگشت به پیش‌فرض",
    # ---------- start screen ----------
    "Rebuild a surface from a point cloud and measure how close it is to "
    "the original": "سطح را از ابر نقاط بازسازی کنید و ببینید چقدر به مدل اصلی نزدیک است",
    "1  Open": "۱  باز کردن",
    "2  Reconstruct": "۲  بازسازی",
    "3  Compare": "۳  مقایسه",
    "A mesh (STL, OBJ, PLY …) or a point cloud. Its surface becomes the "
    "ground truth.": "یک مش (STL، OBJ، PLY و …) یا ابر نقاط. سطح آن مرجع (ground truth) می‌شود.",
    "Clean the points, then build a surface with one of 5 algorithms - or "
    "let the optimizer find the best settings.":
        "نقاط را پاک‌سازی کنید و با یکی از ۵ الگوریتم سطح بسازید، یا بگذارید بهینه‌ساز بهترین تنظیمات را پیدا کند.",
    "Measure the reconstruction against the original: 7 distance methods, "
    "error statistics and colour maps.":
        "بازسازی را با مدل اصلی بسنجید: ۷ روش فاصله، آمار خطا و نقشه‌ی رنگی.",
    "Try the sample bunny": "امتحان با نمونه‌ی خرگوش",
    "or drag a file onto the window": "یا یک فایل را روی پنجره بکشید",
    # ---------- compare tab ----------
    "<b>Reconstruction</b> (compared)": "<b>بازسازی</b> (مقایسه‌شونده)",
    "<b>Reference</b> (original)": "<b>مرجع</b> (مدل اصلی)",
    "Reconstruction (compared)": "بازسازی (مقایسه‌شونده)",
    "Reference (original)": "مرجع (مدل اصلی)",
    "Open reconstruction…": "باز کردن بازسازی…",
    "From Reconstruct tab": "از تب بازسازی",
    "Open reference…": "باز کردن مرجع…",
    "From Import mesh tab": "از تب وارد کردن مش",
    "Align automatically when opening": "هم‌ترازی خودکار هنگام باز کردن",
    "Align now": "هم‌ترازی همین حالا",
    "Samples": "نمونه‌ها",
    "M3C2 cores": "نقاط مرکزی M3C2",
    "M3C2 diameter ×": "قطر M3C2 ×",
    "M3C2 depth": "عمق M3C2",
    "Run comparison": "اجرای مقایسه",
    "Show": "نمایش",
    "One method": "یک روش",
    "All methods (grid)": "همه‌ی روش‌ها (شبکه‌ای)",
    "Overlay the reference (translucent)": "نمایش مرجع روی مدل (نیمه‌شفاف)",
    "Signed colours (blue − / red +)": "رنگ علامت‌دار (آبی − / قرمز +)",
    "Colour max": "حداکثر رنگ",
    "auto (P95)": "خودکار (P95)",
    "Reference opacity": "شفافیت مرجع",
    "Colour the reconstruction by its error": "رنگ‌آمیزی بازسازی بر اساس خطا",
    "Default colours": "رنگ‌های پیش‌فرض",
    "Export results (CSV)…": "ذخیره‌ی نتایج (CSV)…",
    "Copy table as Markdown": "کپی جدول به‌صورت Markdown",
    "All methods overlaid": "همه‌ی روش‌ها روی هم",
    "Selected method only": "فقط روش انتخاب‌شده",
    "Method": "روش",
    "Time": "زمان",
    "Mean e": "میانگین e",
    # ---------- degrade / sweep / report / optimizer dialogs ----------
    "Degrade Data": "تخریب داده",
    "Position noise σ (% of diagonal)": "نویز مکان σ (٪ قطر)",
    "Normal noise σ": "نویز نرمال σ",
    "Holes": "سوراخ‌ها",
    "Hole radius (% of diagonal)": "شعاع سوراخ (٪ قطر)",
    "Non-uniform: keep at far end": "غیریکنواخت: نگه‌داشتن در انتهای محور",
    "Non-uniform: along axis": "غیریکنواخت: در طول محور",
    "Random seed": "بذر تصادفی",
    "Applied after voxel downsampling. A ground-truth mesh that came with "
    "the data stays clean, so the Reconstruction tab measures how robust "
    "each algorithm is.":
        "بعد از کاهش وکسلی اعمال می‌شود. مش مرجعی که همراه داده آمده سالم می‌ماند، پس تب بازسازی می‌سنجد هر الگوریتم چقدر مقاوم است.",
    "Current: clean": "وضعیت فعلی: سالم",
    "Apply": "اعمال",
    "Reset to clean": "بازگشت به داده‌ی سالم",
    "Close": "بستن",
    "Quality Report": "گزارش کیفیت",
    "Reference (ground truth)": "مرجع (ground truth)",
    "Input points (no ground-truth mesh) - measures how well the surface "
    "fits the data.": "نقاط ورودی (بدون مش مرجع) - می‌سنجد سطح چقدر روی داده‌ها می‌نشیند.",
    "Load reference mesh…": "بارگذاری مش مرجع…",
    "Use input points": "استفاده از نقاط ورودی",
    "Evaluate after every reconstruction": "ارزیابی بعد از هر بازسازی",
    "Color mesh by error (heatmap)": "رنگ‌آمیزی مش بر اساس خطا (نقشه‌ی حرارتی)",
    "Robustness sweep…": "آزمون مقاومت…",
    "No evaluation yet.": "هنوز ارزیابی‌ای انجام نشده.",
    "Metric": "معیار",
    "Value": "مقدار",
    "Run history (compare parameter settings):": "تاریخچه‌ی اجراها (مقایسه‌ی تنظیمات):",
    "Reference": "مرجع",
    "Setting": "تنظیم",
    "Data": "داده",
    "Voxel": "وکسل",
    "Density %": "چگالی ٪",
    "Smooth": "هموارسازی",
    "Crop": "برش",
    "Points": "نقاط",
    "Triangles": "مثلث‌ها",
    "Closed": "بسته",
    "Time (s)": "زمان (ثانیه)",
    "Save history CSV…": "ذخیره‌ی تاریخچه (CSV)…",
    "Clear history": "پاک کردن تاریخچه",
    "Robustness Sweep": "آزمون مقاومت",
    "Perturbation": "نوع تخریب",
    "Normal noise σ (degrees)": "نویز نرمال σ (درجه)",
    "Holes (count, radius 5%)": "سوراخ (تعداد، شعاع ۵٪)",
    "Non-uniform sampling: min keep (%)": "نمونه‌برداری غیریکنواخت: حداقل نگه‌داشتن (٪)",
    "Levels (comma separated)": "سطح‌ها (با کاما جدا کنید)",
    "Algorithms": "الگوریتم‌ها",
    "Each level degrades the data loaded in this tab and runs every "
    "selected algorithm with its current parameters.":
        "هر سطح داده‌ی این تب را تخریب می‌کند و هر الگوریتم انتخاب‌شده را با پارامترهای فعلی‌اش اجرا می‌کند.",
    "Run sweep": "اجرای آزمون",
    "Each level degrades the data loaded in this tab and runs every "
    "selected algorithm with its current parameters. Results go to the "
    "Quality Report history. Quality is measured against the reference "
    "mesh, or the clean input points when there is none.":
        "هر سطح، داده‌ی این تب را تخریب می‌کند و هر الگوریتم انتخاب‌شده را با پارامترهای فعلی‌اش اجرا می‌کند. نتایج به تاریخچه‌ی گزارش کیفیت می‌روند. کیفیت با مش مرجع سنجیده می‌شود، یا اگر مرجعی نباشد با نقاط ورودی سالم.",
    "Searches the parameters that fit the <b>reference mesh</b> best "
    "(coordinate descent: a coarse pass per parameter, then refinement "
    "around the best value). The objective is two-sided, so dropping hard "
    "parts of the surface does not pay off.":
        "پارامترهایی را پیدا می‌کند که بهترین تطابق را با <b>مش مرجع</b> دارند (کاهش مختصاتی: یک دور درشت برای هر پارامتر، سپس پالایش اطراف بهترین مقدار). معیار دوطرفه است، پس حذف بخش‌های سخت سطح سودی ندارد.",
    "⚠ No reference mesh: the fit to the input points is optimised (weaker "
    "- load a reference in the Quality report, or send the data from "
    "Import mesh).":
        "⚠ مش مرجع وجود ندارد: تطابق با نقاط ورودی بهینه می‌شود (ضعیف‌تر - یک مرجع در گزارش کیفیت بارگذاری کنید یا داده را از تب وارد کردن مش بفرستید).",
    "Position noise σ (% of diagonal)": "نویز مکان σ (٪ قطر)",
    "Cancel": "لغو",
    "Optimize reconstruction parameters": "بهینه‌سازی پارامترهای بازسازی",
    "All algorithms (fair comparison)": "همه‌ی الگوریتم‌ها (مقایسه‌ی منصفانه)",
    "Objective": "هدف",
    "F-score (higher is better)": "F-score (بیشتر بهتر است)",
    "Chamfer distance (lower is better)": "فاصله‌ی Chamfer (کمتر بهتر است)",
    "Accuracy RMS, reconstruction → reference (lower is better)":
        "RMS دقت، بازسازی ← مرجع (کمتر بهتر است)",
    "Minimum recall": "حداقل recall",
    "Runs per algorithm": "تعداد اجرا برای هر الگوریتم",
    "Check the result on noisy data (σ = 0.25 % of the diagonal)":
        "بررسی نتیجه روی داده‌ی نویزی (σ = ۰٫۲۵٪ قطر)",
    "Run optimisation": "اجرای بهینه‌سازی",
    "Run": "اجرا",
    "Parameters": "پارامترها",
    "Pieces": "تکه‌ها",
    "Apply best to the tab": "اعمال بهترین در تب",
    "Apply && Send to Compare": "اعمال و ارسال به مقایسه",
    "Copy table (Markdown)": "کپی جدول (Markdown)",
    "Save CSV…": "ذخیره‌ی CSV…",
    # ---------- appearance dialogs ----------
    "Object appearance": "ظاهر شیء",
    "Colour": "رنگ",
    "Original colours (from the file; theme colour when it has none)":
        "رنگ‌های اصلی (از فایل؛ اگر نداشت، رنگ تم)",
    "Custom colour": "رنگ دلخواه",
    "Display": "نمایش",
    "Style": "سبک",
    "Surface": "سطح",
    "Surface with edges (mesh lines)": "سطح با یال‌ها (خطوط مش)",
    "Wireframe": "سیمی",
    "Points (vertices only)": "نقاط (فقط رأس‌ها)",
    "Edge colour": "رنگ یال",
    "Opacity": "شفافیت",
    "Point size": "اندازه‌ی نقطه",
    "Save": "ذخیره",
    "Default": "پیش‌فرض",
    "Background and guides": "پس‌زمینه و راهنماها",
    "Theme (light / dark gradient)": "تم (گرادیان روشن / تیره)",
    "Solid colour": "رنگ یکدست",
    "Gradient": "گرادیان",
    "top": "بالا",
    "bottom": "پایین",
    "Grid, scale and reference": "شبکه، مقیاس و مرجع",
    "Floor grid": "شبکه‌ی کف",
    "Ruler (scale on the model's box)": "خط‌کش (مقیاس روی جعبه‌ی مدل)",
    "Object axes": "محورهای شیء",
    "Object box": "جعبه‌ی شیء",
    "Coordinates on the box corners": "مختصات گوشه‌های جعبه",
    "Rotation / position read-out": "نمایش عددی چرخش / مکان",
    "Rotation and position are measured from the pose the model had when "
    "it was loaded.": "چرخش و مکان نسبت به حالتی سنجیده می‌شوند که مدل هنگام بارگذاری داشت.",
    # ---------- menus ----------
    "File": "فایل",
    "Open Recent": "فایل‌های اخیر",
    "Help": "راهنما",
    "Open…": "باز کردن…",
    "Open Sample (Stanford bunny)": "باز کردن نمونه (خرگوش استنفورد)",
    "Export…": "ذخیره…",
    "Save Screenshot…": "ذخیره‌ی تصویر صفحه…",
    "Remove Data from This Tab": "حذف داده‌ی این تب",
    "Remove Data from All Tabs": "حذف داده‌ی همه‌ی تب‌ها",
    "Quit": "خروج",
    "Reset View (rotation, position, zoom)": "بازنشانی نما (چرخش، مکان، بزرگ‌نمایی)",
    "Front View": "نمای جلو",
    "Top View": "نمای بالا",
    "Side View": "نمای کنار",
    "Iso View": "نمای ایزومتریک",
    "Previous Checkpoint": "نقطه‌ی بازگشت قبلی",
    "Next Checkpoint": "نقطه‌ی بازگشت بعدی",
    "Use GPU for Computations": "استفاده از GPU برای محاسبات",
    "Use GPU for Computations (no CUDA GPU found)":
        "استفاده از GPU برای محاسبات (GPU با CUDA پیدا نشد)",
    "Reset Layout": "بازنشانی چیدمان",
    "Pose Bar": "نوار ژست",
    "Mouse Guide": "راهنمای موس",
    "Dark Theme": "تم تیره",
    "Keyboard Shortcuts": "میانبرهای صفحه‌کلید",
    "About": "درباره",
    "Contact…": "تماس…",
    "Contact": "تماس",
    "GitHub and e-mail of the author": "گیت‌هاب و ایمیل سازنده",
    "Questions, bug reports or ideas - get in touch:": "برای سؤال، گزارش مشکل یا پیشنهاد با من در تماس باشید:",
    "Project": "پروژه",
    "E-mail": "ایمیل",
    "Language": "زبان",
    # ---------- tooltips ----------
    "Copy all values as text (tab separated)": "کپی همه‌ی مقدارها به‌صورت متن (جداشده با Tab)",
    "Drag to resize - double-click for the default size":
        "برای تغییر اندازه بکشید - دوبار کلیک برای اندازه‌ی پیش‌فرض",
    "Click to collapse / expand this section\nRight-click: expand or collapse all":
        "کلیک: بستن / باز کردن این بخش\nکلیک راست: باز یا بسته کردن همه",
    "Open the reconstructed mesh (Ctrl+O)": "باز کردن مش بازسازی‌شده (Ctrl+O)",
    "Take the mesh of tab 3 (and its ground truth, if it has one)":
        "گرفتن مش تب ۳ (و مرجعش، اگر داشته باشد)",
    "Open the original mesh (same coordinates as the reconstruction)":
        "باز کردن مش اصلی (در همان مختصات بازسازی)",
    "Use the mesh shown in tab 1 (its rotation applied)": "استفاده از مش تب ۱",
    "Remove the data of this tab and free its memory (Ctrl+W)":
        "حذف داده‌ی این تب و آزاد کردن حافظه (Ctrl+W)",
    "If the two models are not in the same place / orientation (e.g. the "
    "reconstruction was saved rotated), move the reconstruction onto the "
    "reference: feature matching (FPFH + RANSAC) then ICP. Models that "
    "already match are left as they are.":
        "اگر دو مدل در یک مکان یا جهت نباشند (مثلاً بازسازی چرخیده ذخیره شده باشد)، بازسازی روی مرجع منتقل می‌شود: تطبیق ویژگی (FPFH + RANSAC) و سپس ICP. مدل‌هایی که از قبل منطبق‌اند دست نمی‌خورند.",
    "Register the reconstruction onto the reference now (also when they "
    "look aligned: a final ICP fit)":
        "هم‌ترازی بازسازی روی مرجع همین حالا (حتی اگر هم‌تراز به نظر برسند: یک برازش ICP نهایی)",
    "Vertex to the nearest reference vertex": "رأس تا نزدیک‌ترین رأس مرجع",
    "Vertex to the tangent plane of the nearest reference vertex":
        "رأس تا صفحه‌ی مماس نزدیک‌ترین رأس مرجع",
    "Vertex to the exact closest point on the reference triangles - needs "
    "reference triangles": "رأس تا نزدیک‌ترین نقطه‌ی دقیق روی مثلث‌های مرجع - مرجع باید مثلث داشته باشد",
    "Dense surface samples to the nearest reference sample":
        "نمونه‌های چگال سطح تا نزدیک‌ترین نمونه‌ی مرجع",
    "Samples to a local plane fitted to 6-24 reference samples (largest "
    "planar neighbourhood)": "نمونه‌ها تا صفحه‌ی محلی برازش‌شده روی ۶ تا ۲۴ نمونه‌ی مرجع (بزرگ‌ترین همسایگی تخت)",
    "Dense surface samples to the reference triangles - needs reference "
    "triangles": "نمونه‌های چگال سطح تا مثلث‌های مرجع - مرجع باید مثلث داشته باشد",
    "Multiscale model-to-model cloud comparison along local normals, with "
    "a 95% level of detection": "مقایسه‌ی چندمقیاسی دو ابر در امتداد نرمال‌های محلی، با سطح تشخیص ۹۵٪",
    "Points sampled uniformly on each surface for the cloud methods":
        "تعداد نقاطی که برای روش‌های ابری به‌طور یکنواخت روی هر سطح نمونه‌برداری می‌شوند",
    "Points where M3C2 measures a distance": "نقاطی که M3C2 در آن‌ها فاصله را می‌سنجد",
    "M3C2 cylinder diameter as a multiple of the mean sample spacing":
        "قطر استوانه‌ی M3C2 به‌صورت ضریبی از فاصله‌ی میانگین نمونه‌ها",
    "M3C2 maximum depth (cylinder half-length) in % of the diagonal: just "
    "above the expected error - too long and it reaches the other side of "
    "thin parts": "حداکثر عمق M3C2 (نیم‌طول استوانه) به درصد قطر: کمی بیشتر از خطای مورد انتظار؛ اگر خیلی بلند باشد به آن طرف بخش‌های نازک می‌رسد",
    "Compute the checked methods": "محاسبه‌ی روش‌های انتخاب‌شده",
    "One coloured model, or every method side by side with a common colour "
    "scale (the views turn together)": "یک مدل رنگی، یا همه‌ی روش‌ها کنار هم با یک مقیاس رنگ مشترک (نماها با هم می‌چرخند)",
    "Signed methods: blue = inside the reference, red = outside":
        "روش‌های علامت‌دار: آبی = داخل مرجع، قرمز = بیرون",
    "Error shown in full red; 0 = 95th percentile of |e|":
        "خطایی که کاملاً قرمز نشان داده می‌شود؛ ۰ = صدک ۹۵ از |e|",
    "Colour of the reconstruction (shown when it is not coloured by its "
    "error)": "رنگ بازسازی (وقتی بر اساس خطا رنگ‌آمیزی نشده باشد)",
    "Colour of the reference mesh": "رنگ مش مرجع",
    "Opacity of the reference drawn over the reconstruction": "شفافیت مرجعی که روی بازسازی کشیده می‌شود",
    "Off: both models keep their own colours after a comparison":
        "خاموش: بعد از مقایسه هم هر دو مدل رنگ خودشان را دارند",
    "Theme colours, reference 28 % opaque, error colours on":
        "رنگ‌های تم، مرجع با شفافیت ۲۸٪، رنگ خطا روشن",
    "Save the error table (Ctrl+S)": "ذخیره‌ی جدول خطا (Ctrl+S)",
    "Rotation and position back to 0 and fit the model in the view (Ctrl+0)":
        "چرخش و مکان به ۰ برمی‌گردد و مدل در نما قاب می‌شود (Ctrl+0)",
    "No earlier checkpoint": "نقطه‌ی بازگشت قبلی وجود ندارد",
    "Checkpoints appear once data is loaded": "نقاط بازگشت بعد از بارگذاری داده ظاهر می‌شوند",
    "No later checkpoint": "نقطه‌ی بازگشت بعدی وجود ندارد",
    "Look at the model from the front, top, side or at an angle":
        "نگاه به مدل از جلو، بالا، کنار یا از زاویه",
    "Colour and display style of the model": "رنگ و سبک نمایش مدل",
    "Background colour, grid, ruler and start-pose guides": "رنگ پس‌زمینه، شبکه، خط‌کش و راهنماها",
    "Show / hide the rotation and position bar under the view":
        "نمایش / پنهان کردن نوار چرخش و مکان زیر نما",
    "Save the view as a PNG image": "ذخیره‌ی نما به‌صورت تصویر PNG",
    "Left drag anywhere: rotate · right drag: move the model · wheel: zoom":
        "کشیدن با کلیک چپ: چرخش · کلیک راست: جابه‌جایی مدل · چرخ موس: بزرگ‌نمایی",
    "Hide the mouse guide (View > Mouse Guide shows it again)":
        "پنهان کردن راهنمای موس (از منوی نما دوباره نمایش داده می‌شود)",
    "Reset rotation: all angles back to 0°": "بازنشانی چرخش: همه‌ی زاویه‌ها به ۰°",
    "Reset position: move the model back to 0, 0, 0": "بازنشانی مکان: مدل به ۰، ۰، ۰ برمی‌گردد",
    "Move the model along X (model units; you can also drag it with the "
    "right mouse button)": "جابه‌جایی مدل در امتداد X (واحد مدل؛ با کلیک راست موس هم می‌شود کشید)",
    "Move the model along Y (model units; you can also drag it with the "
    "right mouse button)": "جابه‌جایی مدل در امتداد Y (واحد مدل؛ با کلیک راست موس هم می‌شود کشید)",
    "Move the model along Z (model units; you can also drag it with the "
    "right mouse button)": "جابه‌جایی مدل در امتداد Z (واحد مدل؛ با کلیک راست موس هم می‌شود کشید)",
    "Rotation about the X axis in degrees (you can also drag the model "
    "with the left mouse button)": "چرخش حول محور X به درجه (با کلیک چپ موس هم می‌شود مدل را چرخاند)",
    "Rotation about the Y axis in degrees (you can also drag the model "
    "with the left mouse button)": "چرخش حول محور Y به درجه (با کلیک چپ موس هم می‌شود مدل را چرخاند)",
    "Rotation about the Z axis in degrees (you can also drag the model "
    "with the left mouse button)": "چرخش حول محور Z به درجه (با کلیک چپ موس هم می‌شود مدل را چرخاند)",
    "Open a point cloud with normals (Ctrl+O)": "باز کردن ابر نقاط همراه نرمال (Ctrl+O)",
    "Poisson: smooth closed surface, robust to noise - the most accurate "
    "overall.\nBall Pivoting: passes exactly through the points.\nAlpha "
    "Shape: fastest, but many small pieces.\nRIMLS: robust moving least "
    "squares - keeps sharp features, ignores outliers (≈ 10 s).\nSigned "
    "Distance: fast volumetric method, the best choice for noisy data (no "
    "debris).":
        "Poisson: سطح بسته و هموار، مقاوم به نویز - در مجموع دقیق‌ترین.\n"
        "Ball Pivoting: دقیقاً از خود نقاط می‌گذرد.\n"
        "Alpha Shape: سریع‌ترین، ولی با تکه‌های کوچک زیاد.\n"
        "RIMLS: حداقل مربعات متحرک مقاوم - گوشه‌های تیز را نگه می‌دارد و نقاط پرت را نادیده می‌گیرد (حدود ۱۰ ثانیه).\n"
        "Signed Distance: روش حجمی سریع، بهترین انتخاب برای داده‌ی نویزی.",
    "Multiple of the mean distance between neighbouring points.":
        "ضریبی از فاصله‌ی میانگین بین نقاط همسایه.",
    "Search the settings that fit the reference mesh best (one algorithm or "
    "all of them), then apply them": "جستجوی تنظیماتی که بهترین تطابق را با مش مرجع دارند (یک الگوریتم یا همه)، و اعمال آن‌ها",
    "Stop the reconstruction that is running now (Esc); the last finished "
    "mesh stays": "توقف بازسازی در حال اجرا (Esc)؛ آخرین مش تمام‌شده باقی می‌ماند",
    "Octree depth: higher = more detail, slower (8–10 typical)":
        "عمق octree: بیشتر = جزئیات بیشتر و کندتر (معمولاً ۸ تا ۱۰)",
    "Finest octree cell size. 0 = use Depth; a value > 0 overrides Depth.":
        "اندازه‌ی ریزترین سلول octree. ۰ = استفاده از عمق؛ مقدار بزرگ‌تر از ۰ جای عمق را می‌گیرد.",
    "Size of the reconstruction cube relative to the data's bounding box "
    "(1.1 avoids cut edges)": "اندازه‌ی مکعب بازسازی نسبت به جعبه‌ی داده (۱٫۱ از بریده شدن لبه‌ها جلوگیری می‌کند)",
    "Place vertices by linear interpolation (slightly more accurate, "
    "slower)": "تعیین مکان رأس‌ها با درون‌یابی خطی (کمی دقیق‌تر و کندتر)",
    "Remove outliers and orient the normals consistently before "
    "reconstructing": "حذف نقاط پرت و جهت‌دهی سازگار نرمال‌ها پیش از بازسازی",
    "Downsample the input first (0 = keep every point)": "کاهش چگالی ورودی پیش از کار (۰ = همه‌ی نقاط)",
    "Remove the least supported part of the Poisson surface (% of vertices "
    "with the lowest density)": "حذف کم‌پشتیبان‌ترین بخش سطح Poisson (درصد رأس‌هایی با کمترین چگالی)",
    "Cut the Poisson surface to the data's bounding box (removes "
    "'bubbles')": "برش سطح Poisson به جعبه‌ی داده (حذف «حباب‌ها»)",
    "Taubin smoothing iterations (0 = off)": "تعداد تکرار هموارسازی Taubin (۰ = خاموش)",
    "Save the mesh as STL or OBJ, rotation applied (Ctrl+S)":
        "ذخیره‌ی مش به‌صورت STL یا OBJ، با چرخش اعمال‌شده (Ctrl+S)",
    "Quality metrics, error heatmap, run history and robustness sweeps "
    "(Ctrl+R)": "معیارهای کیفیت، نقشه‌ی خطا، تاریخچه‌ی اجراها و آزمون مقاومت (Ctrl+R)",
    "Compare this mesh with the original in tab 4 (7 distance methods, "
    "error statistics)": "مقایسه‌ی این مش با مدل اصلی در تب ۴ (۷ روش فاصله، آمار خطا)",
    "Open a point cloud: PLY, PCD, TXT, CSV or NPY (Ctrl+O)":
        "باز کردن ابر نقاط: PLY، PCD، TXT، CSV یا NPY (Ctrl+O)",
    "Merge all points inside each cube of this size into one point (their "
    "normals are averaged). 0 keeps every point.":
        "همه‌ی نقاط داخل هر مکعب با این اندازه یک نقطه می‌شوند (نرمال‌ها میانگین گرفته می‌شوند). ۰ همه‌ی نقاط را نگه می‌دارد.",
    "Randomise the order of the points (normals move with them)":
        "تصادفی کردن ترتیب نقاط (نرمال‌ها همراهشان جابه‌جا می‌شوند)",
    "Add noise, holes or uneven sampling to test how robust the "
    "reconstruction is": "افزودن نویز، سوراخ یا نمونه‌برداری ناهموار برای سنجش مقاومت بازسازی",
    "Save points + normals (rotation applied) (Ctrl+S)": "ذخیره‌ی نقاط + نرمال‌ها (با چرخش اعمال‌شده) (Ctrl+S)",
    "Reconstruct a surface from this data": "بازسازی سطح از این داده",
    "Open a mesh or point cloud (Ctrl+O)": "باز کردن مش یا ابر نقاط (Ctrl+O)",
    "Save points + normals (PLY / TXT / CSV / NPY), rotation applied "
    "(Ctrl+S)": "ذخیره‌ی نقاط + نرمال‌ها (PLY / TXT / CSV / NPY)، با چرخش اعمال‌شده (Ctrl+S)",
    "Continue in the Processing tab with this data": "ادامه با این داده در تب پردازش",
    "Open the Stanford bunny (Ctrl+Shift+O)": "باز کردن خرگوش استنفورد (Ctrl+Shift+O)",
    "Compare against a mesh in the same coordinates as the data":
        "مقایسه با مشی در همان مختصات داده",
    "Compare against the input points instead of a mesh": "مقایسه با نقاط ورودی به جای مش",
    "Run every algorithm on several levels of noise, holes, …":
        "اجرای همه‌ی الگوریتم‌ها روی چند سطح نویز، سوراخ و …",
    "Copy the table above, ready to paste into a report": "کپی جدول بالا، آماده برای گزارش",
    "Every parameter and metric of every run": "همه‌ی پارامترها و معیارهای هر اجرا",
    "Runs that cover less of the reference (within 1 % of the diagonal) "
    "are not eligible as best - keeps the result complete":
        "اجراهایی که بخش کمتری از مرجع را می‌پوشانند (در فاصله‌ی ۱٪ قطر) نمی‌توانند بهترین باشند - تا نتیجه کامل بماند",
    "Reconstructions tried per algorithm": "تعداد بازسازی‌های آزمایشی برای هر الگوریتم",
    "Re-run the starting and the best settings on a noisy copy: shows "
    "whether the gain is real or only fits this data set":
        "اجرای دوباره‌ی تنظیمات اولیه و بهترین روی یک نسخه‌ی نویزی: نشان می‌دهد بهبود واقعی است یا فقط با همین داده جور است",
    "Set the tab's parameters to the selected row (or the best run) and "
    "rebuild the mesh": "تنظیم پارامترهای تب بر اساس ردیف انتخاب‌شده (یا بهترین اجرا) و ساختن دوباره‌ی مش",
    "How a mesh is drawn (point clouds are always drawn as points)":
        "نحوه‌ی رسم مش (ابر نقاط همیشه به‌صورت نقطه رسم می‌شود)",
    "Fill in the original settings (Save to keep them)": "پر کردن تنظیمات اولیه (برای نگه‌داشتن، ذخیره کنید)",
    "A grid under the model at its starting height; the size of one square "
    "is shown in the read-out": "شبکه‌ای زیر مدل در ارتفاع اولیه‌اش؛ اندازه‌ی هر خانه در نمایش عددی آمده است",
    "Axes with tick labels in model units around the model":
        "محورهای مدرج با واحد مدل در اطراف مدل",
    "The model's own X / Y / Z axes at its centre: they turn and move with "
    "the model, in step with the rotation / position values and the corner "
    "axes": "محورهای X / Y / Z خود مدل در مرکزش: همراه مدل می‌چرخند و جابه‌جا می‌شوند، هماهنگ با مقادیر چرخش و مکان",
    "The model's bounding box, turning and moving with the model":
        "جعبه‌ی محیطی مدل که همراه مدل می‌چرخد و جابه‌جا می‌شود",
    "Writes x, y, z (model units) at each corner of the object box; labels "
    "that would overlap are left out": "مختصات x, y, z (واحد مدل) در هر گوشه‌ی جعبه نوشته می‌شود؛ برچسب‌های هم‌پوشان حذف می‌شوند",
    "Angles and offsets relative to the starting pose, in a corner":
        "زاویه‌ها و جابه‌جایی‌ها نسبت به حالت اولیه، در گوشه‌ی نما",
    "Remove the current tab's data and free its memory": "حذف داده‌ی تب فعلی و آزاد کردن حافظه",
    "Empty every tab and free the memory": "خالی کردن همه‌ی تب‌ها و آزاد کردن حافظه",
    "Nearest-neighbour searches (comparison, quality, RIMLS, …) on the "
    "graphics card when Open3D has CUDA": "جستجوی نزدیک‌ترین همسایه (مقایسه، کیفیت، RIMLS و …) روی کارت گرافیک، اگر Open3D با CUDA باشد",
    "Default sizes of the 3D view, side panel and console": "اندازه‌های پیش‌فرض نمای سه‌بعدی، پنل کناری و کنسول",
    "Show / hide the rotation and position bar under the 3D views":
        "نمایش / پنهان کردن نوار چرخش و مکان زیر نماهای سه‌بعدی",
    "Show / hide the mouse help under the 3D views": "نمایش / پنهان کردن راهنمای موس زیر نماهای سه‌بعدی",
    "Restart the program in this language": "اجرای دوباره‌ی برنامه با این زبان",
    # ---------- object info ----------
    "Property": "ویژگی",
    "Pose": "ژست",
    "Rotation X, Y, Z": "چرخش X، Y، Z",
    "Position X, Y, Z": "مکان X، Y، Z",
    "Object box": "جعبه‌ی شیء",
    "Size (W × H × D)": "اندازه (عرض × ارتفاع × عمق)",
    "Diagonal": "قطر",
    "Volume": "حجم",
    "Centre": "مرکز",
    "World box (axis-aligned)": "جعبه‌ی جهانی (هم‌راستای محورها)",
    "Min X, Y, Z": "کمینه‌ی X، Y، Z",
    "Max X, Y, Z": "بیشینه‌ی X، Y، Z",
    "Size": "اندازه",
    "Count": "تعداد",
    "Centroid": "مرکز ثقل",
    "Mean spacing": "فاصله‌ی میانگین",
    "Min spacing": "کمترین فاصله",
    "Density": "چگالی",
    "Duplicates": "نقاط تکراری",
    "Normals": "نرمال‌ها",
    "Mean length": "طول میانگین",
    "Unit length": "طول واحد",
    "Pointing outward": "رو به بیرون",
    "Mean direction": "جهت میانگین",
    "Coherence": "همسویی",
    "Mesh": "مش",
    "Vertices used": "رأس‌های استفاده‌شده",
    "Edges": "یال‌ها",
    "Surface area": "مساحت سطح",
    "Fills box": "پرشدگی جعبه",
    "Closed (no holes)": "بسته (بدون سوراخ)",
    "Boundary edges": "یال‌های مرزی",
    "Non-manifold edges": "یال‌های غیرمنیفلد",
    "Degenerate triangles": "مثلث‌های منحط",
    "Edge length min / mean / max": "طول یال کمینه / میانگین / بیشینه",
    "Euler characteristic": "مشخصه‌ی اویلر",
    "Processing": "پردازش",
    "Original points": "نقاط اصلی",
    "Kept": "نگه‌داشته‌شده",
    "Voxel size": "اندازه‌ی وکسل",
    "Degradation": "تخریب",
    "Shuffled": "بُر خورده",
    "Reconstruction": "بازسازی",
    "Input points": "نقاط ورودی",
    "Comparison": "مقایسه",
    "Compared": "مقایسه‌شونده",
    "Reference points": "نقاط مرجع",
    "yes": "بله",
    "no": "خیر",
    "none": "ندارد",
    "off": "خاموش",
    "Angles of the model relative to the pose it was loaded in":
        "زاویه‌های مدل نسبت به حالتی که با آن بارگذاری شد",
    "Offset from the starting position (model units)": "فاصله از مکان اولیه (واحد مدل)",
    "Edge lengths of the model's own box (they do not change when it turns)":
        "طول اضلاع جعبه‌ی خود مدل (با چرخش عوض نمی‌شود)",
    "Length of the box diagonal": "طول قطر جعبه",
    "Volume of the box around the model": "حجم جعبه‌ی دور مدل",
    "Surface area of the box": "مساحت سطح جعبه",
    "Centre of the box in world coordinates (moves with the model)":
        "مرکز جعبه در مختصات جهانی (همراه مدل جابه‌جا می‌شود)",
    "Box corner in world coordinates": "گوشه‌ی جعبه در مختصات جهانی",
    "Smallest coordinates in the world": "کوچک‌ترین مختصات در فضای جهانی",
    "Largest coordinates in the world": "بزرگ‌ترین مختصات در فضای جهانی",
    "Extent along the world axes (changes when the model turns)":
        "گستره در امتداد محورهای جهانی (با چرخش مدل عوض می‌شود)",
    "Volume of the axis-aligned box": "حجم جعبه‌ی هم‌راستای محورها",
    "Number of points / vertices": "تعداد نقاط / رأس‌ها",
    "Mean of all points, in world coordinates": "میانگین همه‌ی نقاط، در مختصات جهانی",
    "Average distance to the nearest neighbour": "فاصله‌ی میانگین تا نزدیک‌ترین همسایه",
    "Smallest non-zero distance between neighbours": "کوچک‌ترین فاصله‌ی غیرصفر بین همسایه‌ها",
    "Points per unit volume of the object box": "تعداد نقطه در واحد حجم جعبه‌ی شیء",
    "Points at exactly the same place": "نقاطی که دقیقاً در یک مکان‌اند",
    "1.0000 = normalised": "۱٫۰۰۰۰ = نرمال‌شده",
    "Share of normals with length 1": "سهم نرمال‌های با طول ۱",
    "Share pointing away from the centroid - close to 100% for a "
    "consistently oriented closed shape":
        "سهم نرمال‌هایی که از مرکز ثقل دور می‌شوند - برای شکل بسته‌ی با جهت‌دهی سازگار نزدیک ۱۰۰٪",
    "Average normal direction (world coordinates)": "جهت میانگین نرمال‌ها (مختصات جهانی)",
    "Length of the mean normal: ~0 for a closed shape, 1 when all normals "
    "point the same way": "طول نرمال میانگین: برای شکل بسته حدود ۰، و ۱ وقتی همه‌ی نرمال‌ها یک‌جهت‌اند",
    "Number of faces": "تعداد وجه‌ها",
    "Vertices referenced by faces": "رأس‌هایی که وجه‌ها به آن‌ها ارجاع می‌دهند",
    "Unique edges": "یال‌های یکتا",
    "Sum of triangle areas": "مجموع مساحت مثلث‌ها",
    "Mesh volume / object box volume": "حجم مش / حجم جعبه‌ی شیء",
    "Every edge is shared by exactly two triangles (self-intersections are "
    "not checked)": "هر یال دقیقاً بین دو مثلث مشترک است (خودبرشی بررسی نمی‌شود)",
    "Edges of holes / open borders": "یال‌های سوراخ‌ها / لبه‌های باز",
    "Edges shared by more than two triangles": "یال‌هایی که بین بیش از دو مثلث مشترک‌اند",
    "Zero-area faces": "وجه‌های با مساحت صفر",
    "Lengths of the mesh edges": "طول یال‌های مش",
    "V − E + F: 2 for one closed shape without holes": "V − E + F: برای یک شکل بسته‌ی بدون سوراخ برابر ۲",
    "Points before downsampling / degradation": "نقاط پیش از کاهش چگالی / تخریب",
    "Share of the original points shown": "سهم نقاط اصلی که نمایش داده می‌شود",
    "Downsampling cube size": "اندازه‌ی مکعب کاهش چگالی",
    "Deliberate noise / holes": "نویز / سوراخ عمدی",
    "Point order randomised": "ترتیب نقاط تصادفی شده",
    "Points with normals the mesh was built from": "نقاط همراه نرمالی که مش از آن‌ها ساخته شد",
    "Reconstruction method": "روش بازسازی",
    "Main parameter": "پارامتر اصلی",
    "Total reconstruction time": "کل زمان بازسازی",
    "Quality against the reference (Quality report)": "کیفیت نسبت به مرجع (گزارش کیفیت)",
    "Model coloured by the error": "مدلی که با خطا رنگ‌آمیزی شده",
    "Model compared against": "مدلی که با آن مقایسه می‌شود",
    "Vertices of the reference": "رأس‌های مرجع",
    # ---------- shortcuts / about / status ----------
    "Open a file in the current tab": "باز کردن فایل در تب فعلی",
    "Open the sample Stanford bunny": "باز کردن نمونه‌ی خرگوش استنفورد",
    "Export from the current tab": "ذخیره از تب فعلی",
    "Save a screenshot of the 3D view": "ذخیره‌ی تصویر نمای سه‌بعدی",
    "Remove the current tab's data (frees memory)": "حذف داده‌ی تب فعلی (آزاد کردن حافظه)",
    "Remove the data from all tabs": "حذف داده‌ی همه‌ی تب‌ها",
    "Rotation and position back to 0, fit the view": "چرخش و مکان به ۰، قاب کردن نما",
    "Previous / next checkpoint (10 per tab)": "نقطه‌ی بازگشت قبلی / بعدی (۱۰ تا در هر تب)",
    "Front / Top / Side / Iso view": "نمای جلو / بالا / کنار / ایزومتریک",
    "Quality report": "گزارش کیفیت",
    "Switch light / dark theme": "تعویض تم روشن / تیره",
    "Rotate the model (anywhere in the view)": "چرخاندن مدل (هر جای نما)",
    "Move the model": "جابه‌جایی مدل",
    "Zoom": "بزرگ‌نمایی",
    "Mouse: left drag": "موس: کشیدن با کلیک چپ",
    "Mouse: right / middle drag": "موس: کشیدن با کلیک راست / وسط",
    "Mouse: wheel": "موس: چرخ",
    "You can also drag & drop files onto the window.": "می‌توانید فایل‌ها را روی پنجره هم بکشید و رها کنید.",
    "About MeshBench": "درباره‌ی MeshBench",
    "Version": "نسخه",
    "Created by": "ساخته‌ی",
    "Point-cloud surface reconstruction & comparison": "بازسازی سطح از ابر نقاط و مقایسه",
    "This software and its source code are the property of the author. "
    "Copying, publishing or presenting it, in whole or in part, as someone "
    "else’s work is not permitted; see LICENSE.":
        "این نرم‌افزار و کد منبع آن متعلق به سازنده است. کپی، انتشار یا ارائه‌ی آن، به‌طور کامل یا بخشی از آن، به نام شخص دیگر مجاز نیست؛ به فایل LICENSE مراجعه کنید.",
    "Surface reconstruction from point clouds with Poisson, Ball Pivoting "
    "and Alpha Shapes (Open3D), rendered with VTK.":
        "بازسازی سطح از ابر نقاط با Poisson، Ball Pivoting، Alpha Shape، RIMLS و Signed Distance (Open3D)، با نمایش VTK.",
    "Computations": "محاسبات",
    "Logs": "گزارش‌ها",
    "Nothing to remove - all tabs are empty": "چیزی برای حذف نیست - همه‌ی تب‌ها خالی‌اند",
    "Layout reset to the default sizes": "چیدمان به اندازه‌های پیش‌فرض برگشت",
    "Remove data": "حذف داده",
    "Restart now?": "همین حالا دوباره اجرا شود؟",
}

# texts with numbers / names: (regex, replacement)
PATTERNS = [
    (r"^(\d[\d,]*) reconstructions? will run\.$", r"\1 بازسازی اجرا می‌شود."),
    (r"^Current: (.+)$", r"وضعیت فعلی: \1"),
    (r"^(#[0-9a-fA-F]{6}) \(theme\)$", r"\1 (تم)"),
    (r"^(.+)\nMesh · ([\d,]+) points$", r"\1\nمش · \2 نقطه"),
    (r"^(.+)\nPoint cloud · ([\d,]+) points$", r"\1\nابر نقاط · \2 نقطه"),
    (r"^Opened (.+)$", r"باز شد: \1"),
    (r"^Corner (\d)$", r"گوشه‌ی \1"),
    (r"^✅ Reconstruction done \| vertices: ([\d,]+) \| triangles: ([\d,]+)$",
     r"✅ بازسازی انجام شد | رأس‌ها: \1 | مثلث‌ها: \2"),
    (r"^⏹ Reconstruction stopped - the last finished mesh is kept$",
     r"⏹ بازسازی متوقف شد - آخرین مش تمام‌شده نگه داشته شد"),
    (r"^❌ Reconstruction failed: (.+)$", r"❌ بازسازی ناموفق بود: \1"),
    (r"^Reconstructing \((.+)\)…$", r"در حال بازسازی (\1)…"),
    (r"^Screenshot saved: (.+)$", r"تصویر ذخیره شد: \1"),
    (r"^Loaded (.+)$", r"بارگذاری شد: \1"),
    (r"^🗑 Removed (.+) - memory freed$", r"🗑 حذف شد: \1 - حافظه آزاد شد"),
    (r"^Computations run on: (.+)$", r"محاسبات روی: \1"),
    (r"^(\d+) / (\d+)$", r"\1 / \2"),
]
_COMPILED = [(re.compile(p, re.S), r) for p, r in PATTERNS]
_FOLD = re.compile(r"^([▾▸])(\s+)(.+)$", re.S)

_state = {"language": ENGLISH, "installed": False}


def language():
    return _state["language"]


def tr(text):
    """``text`` in the current language (unchanged when unknown)."""
    if _state["language"] != PERSIAN or not isinstance(text, str) or not text:
        return text
    if text in FA:
        return FA[text]
    folded = _FOLD.match(text)                 # "▾  FILE" (collapsible header)
    if folded:
        return folded.group(1) + folded.group(2) + tr(folded.group(3))
    if "\t" in text:                           # "Front\tCtrl+1"
        label, _, keys = text.partition("\t")
        return tr(label) + "\t" + keys
    if "&" in text:                            # menu mnemonic
        plain = text.replace("&&", "\0").replace("&", "").replace("\0", "&")
        if plain in FA:
            return FA[plain]
        if plain.replace("&", "&&") in FA:
            return FA[plain.replace("&", "&&")]
    for pattern, replacement in _COMPILED:
        if pattern.match(text):
            return pattern.sub(replacement, text)
    return text


# =========================
# Wrapping Qt setters
# =========================
def _wrap(cls, name, positions):
    """Translate the string arguments at ``positions`` of ``cls.name``."""
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        args = list(args)
        for i in positions:
            if i < len(args) and isinstance(args[i], str):
                args[i] = tr(args[i])
        return original(self, *args, **kwargs)
    wrapper.__name__ = name
    setattr(cls, name, wrapper)


def _wrap_first_string(cls, name):
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        args = list(args)
        for i, value in enumerate(args):
            if isinstance(value, str):
                args[i] = tr(value)
                break
        return original(self, *args, **kwargs)
    wrapper.__name__ = name
    setattr(cls, name, wrapper)


def _wrap_static(cls, name, positions):
    original = getattr(cls, name)

    def wrapper(*args, **kwargs):
        args = list(args)
        for i in positions:
            if i < len(args) and isinstance(args[i], str):
                args[i] = tr(args[i])
        return original(*args, **kwargs)
    setattr(cls, name, staticmethod(wrapper))


def _patch_setters():
    w = QtWidgets
    _wrap(w.QLabel, "setText", (0,))
    _wrap(w.QAbstractButton, "setText", (0,))
    _wrap(w.QWidget, "setToolTip", (0,))
    _wrap(w.QWidget, "setStatusTip", (0,))
    _wrap(w.QWidget, "setWindowTitle", (0,))
    _wrap(w.QAction, "setText", (0,))
    _wrap(w.QAction, "setToolTip", (0,))
    _wrap(w.QAction, "setStatusTip", (0,))
    _wrap(w.QGroupBox, "setTitle", (0,))
    original_tree_headers = w.QTreeWidget.setHeaderLabels

    def tree_headers(self, labels):
        return original_tree_headers(self, [tr(label) for label in labels])
    w.QTreeWidget.setHeaderLabels = tree_headers
    _wrap(w.QMenu, "setTitle", (0,))
    _wrap_first_string(w.QMenu, "addAction")
    _wrap_first_string(w.QMenu, "addMenu")
    _wrap_first_string(w.QMenuBar, "addMenu")
    _wrap(w.QTabWidget, "setTabText", (1,))
    _wrap(w.QTabWidget, "setTabToolTip", (1,))
    _wrap_first_string(w.QTabWidget, "addTab")
    _wrap_first_string(w.QComboBox, "addItem")
    _wrap(w.QComboBox, "setItemText", (1,))
    _wrap(w.QSpinBox, "setSpecialValueText", (0,))
    _wrap(w.QDoubleSpinBox, "setSpecialValueText", (0,))
    original_headers = w.QTableWidget.setHorizontalHeaderLabels

    def headers(self, labels):
        return original_headers(self, [tr(label) for label in labels])
    w.QTableWidget.setHorizontalHeaderLabels = headers
    for name in ("warning", "information", "critical", "question"):
        _wrap_static(w.QMessageBox, name, (1, 2))
    _wrap_static(w.QMessageBox, "about", (1, 2))


# =========================
# Texts set in constructors / .ui files
# =========================
# numbers and technical text stay left to right inside the RTL layout
# (otherwise "-0.25" shows as "0.25-" and "(0.4 % of diag.)" is scrambled)
LTR_TYPES = ("QAbstractItemView", "QAbstractSpinBox", "QLineEdit",
             "QPlainTextEdit", "QTextEdit", "QSlider")


def translate_tree(root):
    """Translate the texts of ``root`` and its children in place."""
    w = QtWidgets
    ltr = tuple(getattr(w, name) for name in LTR_TYPES)
    for widget in [root] + root.findChildren(w.QWidget):
        if isinstance(widget, ltr):
            widget.setLayoutDirection(Qt.LeftToRight)
        tip = widget.toolTip()
        if tip:
            widget.setToolTip(tip)              # (the setter translates)
        if widget.isWindow():
            widget.setWindowTitle(widget.windowTitle())
        if isinstance(widget, (w.QLabel, w.QAbstractButton)):
            text = widget.text()
            if text and tr(text) != text:
                widget.setText(text)
        if isinstance(widget, w.QGroupBox):
            widget.setTitle(widget.title())
        if isinstance(widget, w.QComboBox):
            for i in range(widget.count()):
                widget.setItemText(i, widget.itemText(i))
        if isinstance(widget, w.QTabWidget):
            for i in range(widget.count()):
                widget.setTabText(i, widget.tabText(i))
        if isinstance(widget, w.QTableWidget):
            for i in range(widget.columnCount()):
                item = widget.horizontalHeaderItem(i)
                if item is not None:
                    item.setText(tr(item.text()))
        if isinstance(widget, (w.QSpinBox, w.QDoubleSpinBox)):
            special = widget.specialValueText()
            if special:
                widget.setSpecialValueText(special)
        if isinstance(widget, w.QMenu):
            widget.setTitle(widget.title())
    for action in root.findChildren(w.QAction):
        for getter, setter in (("text", "setText"), ("toolTip", "setToolTip"),
                               ("statusTip", "setStatusTip")):
            value = getattr(action, getter)()
            if value and tr(value) != value:
                getattr(action, setter)(value)


class _Translator(QObject):
    """Translates every window the first time it is shown."""

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Show and isinstance(obj, QtWidgets.QWidget)
                and obj.isWindow() and not obj.property("i18n_done")):
            obj.setProperty("i18n_done", True)
            translate_tree(obj)
        return False


def install(app, language_code):
    """Use ``language_code`` for the whole program (call before building
    the windows)."""
    _state["language"] = language_code if language_code in LANGUAGES else ENGLISH
    if _state["language"] != PERSIAN or _state["installed"]:
        return
    _state["installed"] = True
    _patch_setters()
    app.setLayoutDirection(Qt.RightToLeft)
    _state["translator"] = _Translator(app)
    app.installEventFilter(_state["translator"])
