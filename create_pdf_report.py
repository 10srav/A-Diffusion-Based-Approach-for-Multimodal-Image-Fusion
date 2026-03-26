"""
Generate PDF report for Adaptive Diffusion Fusion outputs and results.
Requires: pip install reportlab
"""
import os
import json
import csv
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def get_image_size(img_path, max_width=5*inch, max_height=4*inch):
    """Calculate scaled image dimensions maintaining aspect ratio."""
    with PILImage.open(img_path) as img:
        w, h = img.size
        aspect = w / h
        if w / max_width > h / max_height:
            new_width = max_width
            new_height = max_width / aspect
        else:
            new_height = max_height
            new_width = max_height * aspect
        return new_width, new_height


def create_pdf_report(output_pdf=None, test_results_dir=None):
    """
    Create PDF report summarizing Adaptive Diffusion Fusion results.

    Args:
        output_pdf: Output PDF path (default: output/AdaptiveDiffusionFusion_Report.pdf)
        test_results_dir: Path to test results directory (default: output/test_results)
    """
    if output_pdf is None:
        output_pdf = os.path.join(OUTPUT_DIR, "AdaptiveDiffusionFusion_Report.pdf")
    if test_results_dir is None:
        test_results_dir = os.path.join(OUTPUT_DIR, "test_results")

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
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

    # ---- Title Page ----
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Adaptive Diffusion Fusion", title_style))
    story.append(Paragraph(
        "A Diffusion-Based Approach for<br/>Multimodal Image Fusion",
        ParagraphStyle('Subtitle', parent=styles['Heading2'],
                       fontSize=14, alignment=1, textColor=colors.gray)
    ))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Custom ~17M parameter model trained on M3FD dataset",
        ParagraphStyle('Desc', parent=styles['Normal'],
                       fontSize=10, alignment=1, textColor=colors.gray)
    ))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Inspired by: FusionINV (IEEE TIP 2025)",
        ParagraphStyle('Ref', parent=styles['Normal'],
                       fontSize=10, alignment=1, textColor=colors.gray)
    ))
    story.append(PageBreak())

    # ---- Section 1: Method Overview ----
    story.append(Paragraph("1. Method Overview", heading_style))
    story.append(Paragraph("""
    This implementation uses a custom <b>Adaptive Diffusion Fusion</b> model (~17M parameters)
    trained from scratch on the M3FD dataset. The model learns to fuse infrared and visible images
    through a DDPM/DDIM diffusion framework with content-aware adaptive fusion attention.
    """, body_style))

    story.append(Paragraph("""
    <b>Key Components:</b><br/>
    - <b>Dual Modality Encoders</b>: Separate multi-scale CNNs for IR and VIS (4 resolution levels)<br/>
    - <b>Adaptive Fusion Gates</b>: Conv gating at high-res (256x256, 128x128)<br/>
    - <b>Adaptive Fusion Attention</b>: Cross-attention + gating at low-res (64x64, 32x32)<br/>
    - <b>Denoising U-Net</b>: ~12.5M params with self-attention and timestep embeddings<br/>
    - <b>SDEdit Inference</b>: Refines a pseudo ground-truth via partial noising + denoising
    """, body_style))

    story.append(Paragraph("""
    <b>Training:</b><br/>
    - Dataset: M3FD (200 train / 100 test pairs)<br/>
    - Loss: DDPM MSE with min-SNR weighting (gamma=5.0)<br/>
    - Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)<br/>
    - Schedule: Linear warmup + cosine decay<br/>
    - EMA decay: 0.9999<br/>
    - Mixed precision (AMP) on CUDA
    """, body_style))
    story.append(PageBreak())

    # ---- Section 2: Architecture ----
    story.append(Paragraph("2. Model Architecture", heading_style))

    arch_data = [
        ['Component', 'Details', 'Parameters'],
        ['VIS Modality Encoder', 'Multi-scale CNN [16,32,64,128]', '~2.2M'],
        ['IR Modality Encoder', 'Multi-scale CNN [16,32,64,128]', '~2.2M'],
        ['AdaptiveFusionGate x2', 'Conv gating (256x256, 128x128)', '~0.03M'],
        ['AdaptiveFusionAttention x2', 'Cross-attn + gating (64x64, 32x32)', '~0.3M'],
        ['Denoising U-Net', '[32,64,128,256] + self-attn', '~12.5M'],
        ['Total', '', '~17M'],
    ]

    arch_table = Table(arch_data, colWidths=[2*inch, 2.5*inch, 1*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    story.append(arch_table)
    story.append(Paragraph("Table 1: Model architecture breakdown.", caption_style))

    # ---- Section 3: Fusion Mechanism ----
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("3. Adaptive Fusion Mechanism", heading_style))

    story.append(Paragraph("""
    <b>High-Resolution Scales (AdaptiveFusionGate):</b><br/>
    At 256x256 and 128x128 where cross-attention is too expensive, a simple convolutional
    gating mechanism is used. IR and VIS features are concatenated, passed through a 1x1 conv
    with sigmoid activation, producing a spatially-varying gate:<br/>
    <i>output = gate * IR_feat + (1 - gate) * VIS_feat</i>
    """, body_style))

    story.append(Paragraph("""
    <b>Low-Resolution Scales (AdaptiveFusionAttention):</b><br/>
    At 64x64 and 32x32, bidirectional cross-attention is used. IR queries VIS features
    and VIS queries IR features. A content-aware gating network then blends all four
    representations (IR, VIS, IR-cross, VIS-cross) adaptively:<br/>
    <i>output = gate * (IR + IR_cross) + (1 - gate) * (VIS + VIS_cross)</i>
    """, body_style))
    story.append(PageBreak())

    # ---- Section 4: Test Results (from actual data if available) ----
    story.append(Paragraph("4. Test Results", heading_style))

    summary_file = os.path.join(test_results_dir, "test_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)

        avg = summary.get('average_metrics', {})
        num_images = summary.get('num_images', 0)

        story.append(Paragraph(
            f"Results from M3FD test set ({num_images} images):",
            body_style
        ))

        metric_names = ['MI_IR_F', 'MI_VIS_F', 'SSIM_IR_F', 'SSIM_VIS_F',
                        'CC_IR_F', 'CC_VIS_F', 'EN', 'SD', 'SF', 'AG',
                        'VIF_IR_F', 'VIF_VIS_F']

        metric_labels = {
            'MI_IR_F': 'MI (IR, Fused)',
            'MI_VIS_F': 'MI (VIS, Fused)',
            'SSIM_IR_F': 'SSIM (IR, Fused)',
            'SSIM_VIS_F': 'SSIM (VIS, Fused)',
            'CC_IR_F': 'CC (IR, Fused)',
            'CC_VIS_F': 'CC (VIS, Fused)',
            'EN': 'Entropy',
            'SD': 'Std Deviation',
            'SF': 'Spatial Frequency',
            'AG': 'Avg Gradient',
            'VIF_IR_F': 'VIF (IR, Fused)',
            'VIF_VIS_F': 'VIF (VIS, Fused)',
        }

        table_data = [['Metric', 'Mean', 'Std', 'Min', 'Max']]
        for m in metric_names:
            if m in avg:
                v = avg[m]
                table_data.append([
                    metric_labels.get(m, m),
                    f"{v['mean']:.4f}",
                    f"{v['std']:.4f}",
                    f"{v['min']:.4f}",
                    f"{v['max']:.4f}",
                ])

        if len(table_data) > 1:
            results_table = Table(table_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            story.append(results_table)
            story.append(Paragraph("Table 2: Quality metrics on M3FD test set.", caption_style))
    else:
        story.append(Paragraph(
            "No test results found. Run testing first: <i>python run_test.py</i>",
            body_style
        ))

    # ---- Section 5: Sample Outputs ----
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("5. Sample Outputs", heading_style))

    fused_dir = os.path.join(test_results_dir, "fused")
    if os.path.exists(fused_dir):
        fused_files = sorted(Path(fused_dir).glob("*.png"))[:6]
        if fused_files:
            for i, fused_path in enumerate(fused_files):
                try:
                    w, h = get_image_size(str(fused_path), max_width=5*inch, max_height=3*inch)
                    story.append(Image(str(fused_path), width=w, height=h))
                    story.append(Paragraph(
                        f"Figure {i+1}: {fused_path.stem}",
                        caption_style
                    ))
                    story.append(Spacer(1, 0.1*inch))
                except Exception:
                    pass
        else:
            story.append(Paragraph(
                "No fused images found. Run testing first.",
                body_style
            ))
    else:
        story.append(Paragraph(
            "No fused images directory found. Run testing first: <i>python run_test.py</i>",
            body_style
        ))
    story.append(PageBreak())

    # ---- Section 6: Comparison Table ----
    story.append(Paragraph("6. Comparison with Source Images", heading_style))

    comp_data = [
        ['Characteristic', 'Visible Image', 'Infrared Image', 'Fused Image'],
        ['Color Info', 'Full RGB', 'Grayscale', 'Visible-style RGB'],
        ['Dark Areas', 'Poor visibility', 'Good visibility', 'Good visibility'],
        ['Detail Level', 'High in lit areas', 'Moderate', 'High overall'],
        ['Thermal Targets', 'Not visible', 'Clearly visible', 'Enhanced'],
    ]

    comp_table = Table(comp_data, colWidths=[1.4*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(comp_table)
    story.append(Paragraph("Table 3: Characteristics of source vs. fused images.", caption_style))

    # ---- Section 7: References ----
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("7. References", heading_style))

    story.append(Paragraph("""
    <b>Inspired by:</b> Liang et al., "FusionINV: A Diffusion-Based Approach for Multimodal
    Image Fusion," IEEE Transactions on Image Processing, vol. 34, pp. 5355-5368, 2025.<br/><br/>
    <b>Original Code:</b> https://github.com/erfect2020/FusionINV<br/><br/>
    <b>Dataset:</b> M3FD (Multi-spectral) infrared-visible image pairs<br/><br/>
    <b>Note:</b> This is a custom re-implementation using a lightweight ~17M parameter architecture.
    The original FusionINV uses Stable Diffusion v1.5 (~1.1B params) in a training-free manner.
    """, body_style))

    # Build PDF
    doc.build(story)
    print(f"\nPDF report generated: {output_pdf}")
    return output_pdf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PDF report")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path")
    parser.add_argument("--test_results", type=str, default=None, help="Test results directory")
    args = parser.parse_args()

    create_pdf_report(
        output_pdf=args.output,
        test_results_dir=args.test_results
    )
