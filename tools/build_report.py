"""Render the supplied English report and inspectable page PNGs.

Authoring-only dependencies are loaded from the existing environment or
read-only local uv wheel cache. This utility never installs a package.
"""
import argparse
import html
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
import textwrap


def existing_dependencies():
    cache = Path.home() / 'AppData/Local/uv/cache/archive-v0'
    for package in ('charset_normalizer', 'PIL', 'reportlab', 'pypdf', 'pypdfium2'):
        if importlib.util.find_spec(package) is None and cache.exists():
            for child in sorted(cache.iterdir()):
                if (child / package / '__init__.py').is_file():
                    sys.path.insert(0, str(child))
                    break
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f'Report-only dependency unavailable: {package}; see report/requirements-report.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('report/REPORT.md'))
    parser.add_argument('--output', type=Path, default=Path('output/pdf/MP2_Report.pdf'))
    parser.add_argument('--preview-dir', type=Path, default=Path('tmp/pdfs/mp2-report'))
    parser.add_argument('--freeze', type=Path, default=Path('report/RESULTS.json'),
                        help='Report evidence JSON containing version_identifier')
    args = parser.parse_args()
    existing_dependencies()
    from PIL import Image, ImageDraw
    import pypdf
    import pypdfium2 as pdfium
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted

    windows = Path('C:/Windows/Fonts')
    linux = Path('/usr/share/fonts/truetype/dejavu')
    for name, win, lin in (
        ('Body', 'arial.ttf', 'DejaVuSans.ttf'),
        ('BodyBold', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'),
        ('Mono', 'consola.ttf', 'DejaVuSansMono.ttf'),
    ):
        path = windows / win if (windows / win).exists() else linux / lin
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    navy, teal, light = colors.HexColor('#173149'), colors.HexColor('#087F8C'), colors.HexColor('#EEF3F7')
    width = A4[0] - 88
    styles = {
        'body': ParagraphStyle('Body', fontName='Body', fontSize=10.1, leading=12.7, spaceAfter=4.5, alignment=TA_LEFT),
        'compact': ParagraphStyle('Compact', fontName='Body', fontSize=9.6, leading=12.0, spaceAfter=4.0, alignment=TA_LEFT),
        'title': ParagraphStyle('Title', fontName='BodyBold', fontSize=18.5, leading=22, textColor=navy, spaceAfter=10, keepWithNext=True),
        'h2': ParagraphStyle('H2', fontName='BodyBold', fontSize=13.2, leading=16, textColor=navy, spaceBefore=4, spaceAfter=8, keepWithNext=True),
        'h3': ParagraphStyle('H3', fontName='BodyBold', fontSize=10.8, leading=14, textColor=teal, spaceBefore=7, spaceAfter=5, keepWithNext=True),
        'table': ParagraphStyle('Cell', fontName='Body', fontSize=8.4, leading=10.4, spaceAfter=0),
        'tablehead': ParagraphStyle('Head', fontName='BodyBold', fontSize=8.4, leading=10.5, textColor=colors.white),
        'code': ParagraphStyle('Code', fontName='Mono', fontSize=7.8, leading=10.0, leftIndent=7, spaceBefore=4, spaceAfter=8),
    }

    def inline(text):
        text = html.escape(text, quote=False)
        text = re.sub(r'`([^`]+)`', r'<font name="Mono">\1</font>', text)
        return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)

    story = []
    body_style = styles['body']
    lines = args.source.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- PAGEBREAK -->':
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith('```'):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.extend(textwrap.wrap(lines[i], width=104, subsequent_indent='    ',
                                          break_long_words=False, break_on_hyphens=False) or [''])
                i += 1
            story.append(Preformatted('\n'.join(code), styles['code']))
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', cell) for cell in row):
                    rows.append(row)
                i += 1
            n = len(rows[0])
            col_widths = [width * 0.44] + [width * 0.56 / (n - 1)] * (n - 1) if n > 2 else [width * 0.43, width * 0.57]
            data = [[Paragraph(inline(cell), styles['tablehead'] if r == 0 else styles['table'])
                     for cell in row] for r, row in enumerate(rows)]
            table = Table(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), navy), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light]),
                ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LINEBELOW', (0, -1), (-1, -1), 0.4, colors.HexColor('#CCD7DF')),
            ]))
            story.extend([table, Spacer(1, 9)])
            continue
        heading = re.match(r'^(#{1,3})\s+(.+)', line)
        if heading:
            if len(heading[1]) == 2 and heading[2].startswith('5.'):
                body_style = styles['compact']
            style = styles['title' if len(heading[1]) == 1 else 'h2' if len(heading[1]) == 2 else 'h3']
            story.append(Paragraph(inline(heading[2]), style))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '|', '```', '<!-- PAGEBREAK')):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(' '.join(paragraph)), body_style))

    manifest = json.loads(args.freeze.read_text())
    version = manifest['version_identifier']

    def page_frame(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(teal)
        canvas.setLineWidth(1.4)
        canvas.line(44, A4[1] - 31, A4[0] - 44, A4[1] - 31)
        canvas.setFillColor(navy)
        canvas.setFont('Body', 8)
        canvas.drawString(44, A4[1] - 23, 'DASE7506  |  MP2  |  Candidate selection')
        canvas.setFillColor(colors.HexColor('#536779'))
        canvas.drawString(44, 26, version + '  |  Submitted artefact and evaluation')
        canvas.drawRightString(A4[0] - 44, 26, str(doc.page))
        canvas.restoreState()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=A4, leftMargin=44, rightMargin=44,
                            topMargin=46, bottomMargin=43, title='DASE7506 MP2: Reliable Candidate Selection',
                            author='Anonymous student submission')
    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    reader = pypdf.PdfReader(args.output)
    if len(reader.pages) > 10:
        raise ValueError(f'Report exceeds ten pages: {len(reader.pages)}')
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    document = pdfium.PdfDocument(str(args.output))
    for page_index in range(len(document)):
        page = document[page_index]
        bitmap = page.render(scale=1.5)
        img = bitmap.to_pil()
        path = args.preview_dir / f'page-{page_index + 1:02d}.png'
        img.save(path)
        previews.append(path)
        bitmap.close()
        page.close()
    document.close()
    tiles = []
    for index, path in enumerate(previews, 1):
        img = Image.open(path).convert('RGB')
        img.thumbnail((360, 510))
        tile = Image.new('RGB', (380, 540), 'white')
        tile.paste(img, ((380 - img.width) // 2, 18))
        ImageDraw.Draw(tile).text((12, 520), f'Page {index}', fill='black')
        tiles.append(tile)
    sheet = Image.new('RGB', (760, 540 * ((len(tiles) + 1) // 2)), '#CBD5DF')
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 380, (index // 2) * 540))
    sheet.save(args.preview_dir / 'contact-sheet.png')
    versions = {name: importlib.metadata.version(name) for name in ('reportlab', 'pypdf', 'pypdfium2', 'Pillow')}
    qa = {'pages': len(reader.pages), 'page_limit': 10, 'pdf': str(args.output),
          'previews': [str(p) for p in previews], 'libraries': versions,
          'page_text_lengths': [len(page.extract_text()) for page in reader.pages]}
    (args.preview_dir / 'qa.json').write_text(json.dumps(qa, indent=2) + '\n')
    Path('report/requirements-report.txt').write_text(''.join(f'{name}=={version}\n' for name, version in versions.items()))
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
