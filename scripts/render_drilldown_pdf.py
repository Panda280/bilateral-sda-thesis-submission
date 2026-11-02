"""Render prepared drill-down content using the bundled ReportLab runtime."""
from pathlib import Path
import json, argparse
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[3]
B=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/full_drilldown_20260910'
OUT=ROOT/'out/hierarchical_drilldown.pdf'
ap=argparse.ArgumentParser()
ap.add_argument('--content-dir',type=Path,default=B)
ap.add_argument('--output',type=Path,default=OUT)
ap.add_argument('--footer',default='Conditional GWP100 analysis | R-2026-055/056 | 10 September 2026')
ap.add_argument('--title',default='Full hierarchical drill-down: sourcing and intensity')
args=ap.parse_args();B=args.content_dir;OUT=args.output
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold',r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='Arial',fontSize=10,leading=14.5,spaceAfter=9,textColor=colors.HexColor('#25313c')))
styles.add(ParagraphStyle(name='TitleReport',fontName='Arial-Bold',fontSize=19,leading=23,spaceAfter=16,textColor=colors.HexColor('#173c55')))
styles.add(ParagraphStyle(name='CellReport',fontName='Arial',fontSize=8.2,leading=10.8))
styles.add(ParagraphStyle(name='HeaderReport',fontName='Arial-Bold',fontSize=8.2,leading=10.8,textColor=colors.white))
width=A4[0]-88
story=[];pages=json.loads((B/'report_content.json').read_text(encoding='utf-8'))
for i,pg in enumerate(pages):
 if i:story.append(PageBreak())
 story.append(Paragraph(escape(pg['title']),styles['TitleReport']))
 for item in pg['items']:
  if item['type']=='paragraph':story.append(Paragraph(item['text'],styles['BodyReport']))
  elif item['type']=='image':
   from PIL import Image as PILImage
   with PILImage.open(item['path']) as im:aspect=im.width/im.height
   h=min(item['height'],width/aspect);story.extend([Image(item['path'],width=h*aspect,height=h),Spacer(1,10)])
  elif item['type']=='table':
   data=[[Paragraph(escape(str(x)),styles['HeaderReport']) for x in item['headers']]]
   data.extend([[Paragraph(escape(str(x)),styles['CellReport']) for x in row] for row in item['rows']])
   widths=[width*x for x in item['widths']] if item['widths'] else None
   t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
   t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173c55')),('VALIGN',(0,0),(-1,-1),'TOP'),
    ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4f6'),colors.white]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#93a6b4'))]))
   story.extend([t,Spacer(1,12)])
def footer(canvas,doc):
 canvas.saveState();canvas.setFont('Arial',7);canvas.setFillColor(colors.HexColor('#536675'))
 canvas.drawString(44,24,args.footer)
 canvas.drawRightString(A4[0]-44,24,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=42,title=args.title,author='Bram Honig')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
