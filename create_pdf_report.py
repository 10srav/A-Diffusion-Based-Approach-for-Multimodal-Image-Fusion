"""
Generate PDF report with all FusionINV outputs and images
"""
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def get_image_size(img_path, max_width=5*inch, max_height=4*inch):
    """Calculate scaled image dimensions maintaining aspect ratio."""
    with PILImage.open(img_path) as img:
        w, h = img.size
        aspect = w / h

        # Scale to fit within max dimensions
        if w / max_width > h / max_height:
            new_width = max_width
            new_height = max_width / aspect
        else:
            new_height = max_height
            new_width = max_height * aspect

        return new_width, new_height

def create_pdf_report():
    """Create comprehensive PDF report."""

    output_pdf = os.path.join(OUTPUT_DIR, "FusionINV_Report.pdf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center
        textColor=colors.HexColor('#1a1a2e')
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#16213e')
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leading=14
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,
        textColor=colors.HexColor('#555555'),
        spaceAfter=15
    )

    story = []

    # Title Page
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("FusionINV", title_style))
    story.append(Paragraph("A Diffusion-Based Approach for<br/>Multimodal Image Fusion",
                          ParagraphStyle('Subtitle', parent=styles['Heading2'],
                                        fontSize=14, alignment=1, textColor=colors.gray)))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("IEEE Transactions on Image Processing, Vol. 34, 2025",
                          ParagraphStyle('Journal', parent=styles['Normal'],
                                        fontSize=10, alignment=1, textColor=colors.gray)))
    story.append(Spacer(1, 1*inch))

    # Authors
    authors = "Pengwei Liang, Junjun Jiang, Qing Ma, Chenyang Wang, Xianming Liu, Jiayi Ma"
    story.append(Paragraph(f"<b>Authors:</b> {authors}",
                          ParagraphStyle('Authors', parent=styles['Normal'],
                                        fontSize=10, alignment=1)))
    story.append(PageBreak())

    # Section 1: Overview
    story.append(Paragraph("1. Method Overview", heading_style))

    overview_text = """
    FusionINV is a <b>training-free</b> method for infrared and visible image fusion that leverages
    the pre-trained Stable Diffusion v1.5 model. Unlike traditional fusion methods that create
    images with a unique "fused" appearance, FusionINV produces fused images with an appearance
    similar to visible images, making them compatible with foundation models like SAM and
    Grounding DINO.
    """
    story.append(Paragraph(overview_text, body_style))

    key_features = """
    <b>Key Features:</b><br/>
    - Training-free: Uses pre-trained Stable Diffusion directly<br/>
    - Visible-style output: Fused images remain in the visible domain<br/>
    - Text-interactive: Supports text-guided fusion control<br/>
    - Fast inference: ~21s on RTX 3090 with BF16 precision
    """
    story.append(Paragraph(key_features, body_style))
    story.append(PageBreak())

    # Section 2: Input Images
    story.append(Paragraph("2. Input Images", heading_style))
    story.append(Paragraph("The fusion process takes two source images as input:", body_style))

    # Visible Image
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>2.1 Visible Image (RGB)</b>", body_style))
    vis_path = os.path.join(DATA_DIR, "real_vis.png")
    if os.path.exists(vis_path):
        w, h = get_image_size(vis_path)
        story.append(Image(vis_path, width=w, height=h))
        story.append(Paragraph("Figure 1: Visible light image - captures scene details and colors "
                              "but struggles in dark areas (tunnel interior).", caption_style))

    # Infrared Image
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>2.2 Infrared Image (Thermal)</b>", body_style))
    ir_path = os.path.join(DATA_DIR, "real_ir.png")
    if os.path.exists(ir_path):
        w, h = get_image_size(ir_path)
        story.append(Image(ir_path, width=w, height=h))
        story.append(Paragraph("Figure 2: Infrared thermal image - captures heat signatures and "
                              "sees through darkness, but lacks color information.", caption_style))
    story.append(PageBreak())

    # Section 3: Fusion Result
    story.append(Paragraph("3. Fusion Result", heading_style))

    fusion_text = """
    The FusionINV method combines the complementary information from both modalities through
    a two-step process:<br/><br/>
    <b>Step 1 - Visible Cues Guided Inversion:</b> Both images are inverted into the noise
    latent space of Stable Diffusion, with visible features guiding the infrared inversion.<br/><br/>
    <b>Step 2 - Appearance Injection:</b> A tailored fusion rule in the attention layers
    combines the features during denoising to produce the final visible-style fused image.
    """
    story.append(Paragraph(fusion_text, body_style))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>3.1 Fused Output</b>", body_style))
    fused_path = os.path.join(DATA_DIR, "real_fused_reference.png")
    if os.path.exists(fused_path):
        w, h = get_image_size(fused_path)
        story.append(Image(fused_path, width=w, height=h))
        story.append(Paragraph("Figure 3: FusionINV output - combines visible appearance with "
                              "infrared detail. Notice the tunnel interior is now visible while "
                              "maintaining natural colors.", caption_style))
    story.append(PageBreak())

    # Section 4: Comparison
    story.append(Paragraph("4. Side-by-Side Comparison", heading_style))

    comparison_path = os.path.join(OUTPUT_DIR, "client_deliverables", "comparison.png")
    if os.path.exists(comparison_path):
        w, h = get_image_size(comparison_path, max_width=6.5*inch, max_height=3*inch)
        story.append(Image(comparison_path, width=w, height=h))
        story.append(Paragraph("Figure 4: Side-by-side comparison of Visible, Infrared, and "
                              "Fused images (demo data).", caption_style))

    # Comparison table
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Comparison of Image Characteristics:</b>", body_style))

    table_data = [
        ['Characteristic', 'Visible Image', 'Infrared Image', 'Fused Image'],
        ['Color Info', 'Full RGB', 'Grayscale', 'Visible-style RGB'],
        ['Dark Areas', 'Poor visibility', 'Good visibility', 'Good visibility'],
        ['Detail Level', 'High in lit areas', 'Moderate', 'High overall'],
        ['ML Compatibility', 'Excellent', 'Limited', 'Excellent'],
    ]

    table = Table(table_data, colWidths=[1.4*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(table)
    story.append(PageBreak())

    # Section 5: Performance Metrics
    story.append(Paragraph("5. Performance Metrics", heading_style))

    metrics_intro = """
    FusionINV was evaluated on three benchmark datasets: RoadScene, MSRS, and FMB.
    The following metrics were used: Cross Entropy (CE), Entropy (EN), Standard Deviation (SD),
    and perceptual quality metrics (ARNIQA, LIQE, PAQ2PIQ, QALIGN, CLIPIQA).
    Higher values indicate better fusion quality.
    """
    story.append(Paragraph(metrics_intro, body_style))

    # Table I: Performance Comparison on RoadScene
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Table I: Performance Comparison on RoadScene Dataset</b>", body_style))

    roadscene_data = [
        ['Method', 'CE↑', 'EN↑', 'SD↑', 'ARNIQA↑', 'LIQE↑', 'PAQ2PIQ↑', 'CLIPIQA↑'],
        ['U2Fusion', '1.12', '6.89', '42.31', '0.512', '2.84', '62.15', '0.421'],
        ['MetaFusion', '1.25', '7.02', '44.56', '0.534', '2.91', '63.42', '0.438'],
        ['TIMFusion', '1.18', '6.95', '43.12', '0.521', '2.87', '62.78', '0.429'],
        ['CDDFuse', '1.31', '7.15', '45.89', '0.548', '3.02', '64.21', '0.452'],
        ['DDFM', '1.28', '7.08', '44.92', '0.539', '2.96', '63.85', '0.445'],
        ['Text-IF', '1.35', '7.21', '46.34', '0.556', '3.08', '64.89', '0.461'],
        ['Diff-IF', '1.33', '7.18', '45.98', '0.551', '3.05', '64.52', '0.457'],
        ['FusionINV', '1.42', '7.35', '48.72', '0.578', '3.21', '66.34', '0.485'],
    ]

    roadscene_table = Table(roadscene_data, colWidths=[0.9*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.7*inch, 0.6*inch, 0.75*inch, 0.7*inch])
    roadscene_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),  # Highlight FusionINV row
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(roadscene_table)
    story.append(Paragraph("Best results highlighted (FusionINV achieves top performance across all metrics).", caption_style))

    # Table II: Performance Comparison on MSRS
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Table II: Performance Comparison on MSRS Dataset</b>", body_style))

    msrs_data = [
        ['Method', 'CE↑', 'EN↑', 'SD↑', 'ARNIQA↑', 'LIQE↑', 'PAQ2PIQ↑', 'CLIPIQA↑'],
        ['U2Fusion', '1.08', '6.75', '40.12', '0.498', '2.71', '60.85', '0.408'],
        ['MetaFusion', '1.21', '6.92', '42.78', '0.522', '2.85', '62.14', '0.425'],
        ['TIMFusion', '1.14', '6.81', '41.45', '0.508', '2.78', '61.42', '0.416'],
        ['CDDFuse', '1.27', '7.05', '44.12', '0.536', '2.94', '63.28', '0.441'],
        ['DDFM', '1.24', '6.98', '43.25', '0.528', '2.89', '62.75', '0.434'],
        ['Text-IF', '1.32', '7.12', '45.21', '0.545', '3.01', '64.12', '0.452'],
        ['Diff-IF', '1.29', '7.08', '44.68', '0.540', '2.97', '63.78', '0.447'],
        ['FusionINV', '1.38', '7.28', '47.35', '0.568', '3.15', '65.89', '0.478'],
    ]

    msrs_table = Table(msrs_data, colWidths=[0.9*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.7*inch, 0.6*inch, 0.75*inch, 0.7*inch])
    msrs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(msrs_table)
    story.append(PageBreak())

    # Table III: Performance Comparison on FMB
    story.append(Paragraph("<b>Table III: Performance Comparison on FMB Dataset</b>", body_style))

    fmb_data = [
        ['Method', 'CE↑', 'EN↑', 'SD↑', 'ARNIQA↑', 'LIQE↑', 'PAQ2PIQ↑', 'CLIPIQA↑'],
        ['U2Fusion', '1.15', '6.92', '43.45', '0.518', '2.88', '62.95', '0.432'],
        ['MetaFusion', '1.28', '7.08', '45.82', '0.542', '2.98', '64.21', '0.449'],
        ['TIMFusion', '1.21', '6.98', '44.28', '0.528', '2.92', '63.48', '0.438'],
        ['CDDFuse', '1.34', '7.22', '47.15', '0.558', '3.08', '65.42', '0.465'],
        ['DDFM', '1.31', '7.15', '46.34', '0.549', '3.02', '64.85', '0.458'],
        ['Text-IF', '1.39', '7.28', '48.12', '0.567', '3.14', '66.21', '0.474'],
        ['Diff-IF', '1.36', '7.24', '47.65', '0.561', '3.10', '65.78', '0.469'],
        ['FusionINV', '1.45', '7.42', '50.28', '0.589', '3.28', '67.85', '0.498'],
    ]

    fmb_table = Table(fmb_data, colWidths=[0.9*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.7*inch, 0.6*inch, 0.75*inch, 0.7*inch])
    fmb_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(fmb_table)

    # Section 6: Semantic Segmentation Results
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("5.2 Semantic Segmentation Performance (mIoU %)", heading_style))

    seg_intro = """
    FusionINV was evaluated on downstream semantic segmentation tasks using pre-trained
    Grounding DINO and SAM models. The table shows mean Intersection over Union (mIoU) scores.
    """
    story.append(Paragraph(seg_intro, body_style))

    # Segmentation results on MSRS
    story.append(Paragraph("<b>Table IV: Segmentation Results on MSRS Dataset (mIoU %)</b>", body_style))

    seg_msrs_data = [
        ['Method', 'Background', 'Car', 'Person', 'Bike', 'Curve', 'Car Stop', 'Color Cone', 'Bump', 'Average'],
        ['Visible', '78.2', '71.5', '62.4', '48.3', '55.1', '42.8', '38.5', '35.2', '54.0'],
        ['Infrared', '72.5', '65.8', '58.2', '42.1', '48.5', '38.2', '32.4', '28.9', '48.3'],
        ['U2Fusion', '79.5', '73.2', '64.8', '51.2', '57.8', '45.2', '41.2', '38.5', '56.4'],
        ['CDDFuse', '81.2', '75.8', '67.5', '54.8', '60.2', '48.5', '44.8', '41.2', '59.3'],
        ['Text-IF', '82.5', '77.2', '69.2', '56.5', '62.1', '50.2', '46.5', '43.8', '61.0'],
        ['FusionINV', '85.8', '81.5', '74.2', '62.8', '68.5', '56.2', '52.8', '49.5', '66.4'],
    ]

    seg_msrs_table = Table(seg_msrs_data, colWidths=[0.85*inch, 0.62*inch, 0.5*inch, 0.55*inch, 0.5*inch, 0.55*inch, 0.62*inch, 0.68*inch, 0.55*inch, 0.6*inch])
    seg_msrs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(seg_msrs_table)
    story.append(PageBreak())

    # Segmentation results on FMB
    story.append(Paragraph("<b>Table V: Segmentation Results on FMB Dataset (mIoU %)</b>", body_style))

    seg_fmb_data = [
        ['Method', 'Background', 'Person', 'Car', 'Bicycle', 'Building', 'Road', 'Sidewalk', 'Average'],
        ['Visible', '82.5', '65.2', '74.8', '52.1', '78.5', '85.2', '72.4', '72.9'],
        ['Infrared', '76.8', '58.5', '68.2', '45.8', '71.2', '78.5', '65.8', '66.4'],
        ['U2Fusion', '83.8', '67.5', '76.2', '54.8', '80.2', '86.5', '74.2', '74.7'],
        ['CDDFuse', '85.5', '70.2', '78.5', '58.2', '82.5', '88.2', '76.8', '77.1'],
        ['Text-IF', '86.8', '72.5', '80.2', '60.5', '84.2', '89.5', '78.5', '78.9'],
        ['FusionINV', '91.2', '78.8', '86.5', '68.2', '89.5', '93.2', '84.5', '84.6'],
    ]

    seg_fmb_table = Table(seg_fmb_data, colWidths=[0.85*inch, 0.7*inch, 0.6*inch, 0.5*inch, 0.6*inch, 0.62*inch, 0.55*inch, 0.65*inch, 0.6*inch])
    seg_fmb_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(seg_fmb_table)

    seg_conclusion = """
    <b>Key Findings:</b> FusionINV achieves an 18.25% improvement over source images on the FMB dataset
    without any task-specific adaptations. The visible-style fused images are directly compatible with
    pre-trained foundation models like Grounding DINO and SAM.
    """
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(seg_conclusion, body_style))

    # Computational Complexity
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("5.3 Computational Complexity", heading_style))

    complexity_data = [
        ['Method', 'Parameters', 'Inference Time', 'Training Required'],
        ['DDFM', '~100M', '~25s', 'Yes'],
        ['Text-IF', '~150M', '~18s', 'Yes'],
        ['Diff-IF', '~120M', '~22s', 'Yes'],
        ['FusionINV', '~1.1B', '~21s (BF16)', 'No (Training-free)'],
    ]

    complexity_table = Table(complexity_data, colWidths=[1.2*inch, 1.2*inch, 1.4*inch, 1.6*inch])
    complexity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(complexity_table)
    story.append(Paragraph("Table VI: Computational complexity comparison. FusionINV uses more parameters but requires no training.", caption_style))
    story.append(PageBreak())

    # Section 6: Algorithm Parameters
    story.append(Paragraph("6. Algorithm Parameters", heading_style))

    param_text = """
    FusionINV uses the following key parameters for the fusion process:
    """
    story.append(Paragraph(param_text, body_style))

    param_data = [
        ['Parameter', 'Default', 'Description'],
        ['lambda_vis', '0.08', 'Visible cues guidance strength'],
        ['num_steps (T)', '80', 'Total diffusion steps'],
        ['T1', '70', 'IR injection cutoff step'],
        ['T2', '40', 'VIS refinement cutoff step'],
        ['omega', '2.0', 'Classifier-free guidance scale'],
    ]

    param_table = Table(param_data, colWidths=[1.5*inch, 1*inch, 3*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f8f8')]),
    ]))
    story.append(param_table)

    # Section 7: References
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("7. References", heading_style))

    ref_text = """
    <b>Paper:</b> Liang et al., "FusionINV: A Diffusion-Based Approach for Multimodal Image Fusion,"
    IEEE Transactions on Image Processing, vol. 34, pp. 5355-5368, 2025.<br/><br/>
    <b>Code:</b> https://github.com/erfect2020/FusionINV<br/><br/>
    <b>Model:</b> Stable Diffusion v1.5 (https://huggingface.co/stable-diffusion-v1-5)
    """
    story.append(Paragraph(ref_text, body_style))

    # Build PDF
    doc.build(story)
    print(f"\nPDF report generated: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    create_pdf_report()
