"""Generates the sample PDFs (run once):  python scripts/make_sample_pdfs.py"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
styles = getSampleStyleSheet()


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.drawString(50, 28, "Northwind Labs - Confidential")
    canvas.drawRightString(545, 28, f"Page {doc.page}")


def build(name, story):
    SimpleDocTemplate(str(OUT / name), pagesize=A4, title=name).build(story, onFirstPage=footer, onLaterPages=footer)


H, B = styles["Heading2"], styles["BodyText"]
build("security_handbook.pdf", [
    Paragraph("Information Security Handbook", styles["Title"]),
    Paragraph("1. Passwords and Authentication", H),
    Paragraph("All passwords must be at least 14 characters long and must not reuse any of the previous 10 passwords. "
              "Passwords must be rotated every 180 days. Multi-factor authentication (MFA) is mandatory for all company "
              "accounts, including email, the code repository and the cloud console. Hardware security keys are preferred "
              "over SMS codes. Shared accounts are not permitted.", B),
    PageBreak(),
    Paragraph("2. Incident Reporting", H),
    Paragraph("Any suspected security incident, including lost devices, phishing emails that were clicked, or unauthorised "
              "access, must be reported to the Security Team within 24 hours of discovery. Report incidents through the "
              "security hotline or by emailing the incident mailbox. Do not attempt to investigate or delete evidence "
              "yourself. The Security Team will acknowledge every report within 2 hours during business days.", B),
    Paragraph("3. Data Classification", H),
    Paragraph("Company data is classified as Public, Internal, Confidential or Restricted. Restricted data, such as customer "
              "payment information, must be encrypted at rest and in transit and may only be accessed by named individuals. "
              "Confidential data may not be stored on personal devices.", B),
    PageBreak(),
    Paragraph("4. Device Security", H),
    Paragraph("Company laptops must use full-disk encryption and lock automatically after 5 minutes of inactivity. "
              "Operating system updates must be installed within 14 days of release. Personal USB storage devices must not "
              "be connected to company laptops.", B),
    Paragraph("5. Remote Work Security", H),
    Paragraph("When working from public places such as cafes or airports, employees must connect through the company VPN. "
              "Screens must not be visible to bystanders, and confidential calls should not be taken in public. Home routers "
              "should use WPA3 or WPA2 encryption with a non-default administrator password.", B),
    Paragraph("6. Third-Party Vendors", H),
    Paragraph("Vendors that process Confidential or Restricted data must complete a security review before contract signature. "
              "The review is renewed every 12 months. Vendors must notify Northwind Labs of any breach affecting our data "
              "within 72 hours.", B),
])

table = Table([["Metric", "Q2", "Q3"], ["Revenue (USD)", "3.7 million", "4.2 million"], ["Quarter-on-quarter growth", "9%", "12%"],
               ["Headcount", "81", "87"], ["Customer churn", "2.4%", "1.9%"]], hAlign="LEFT")
table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
build("quarterly_report_q3.pdf", [
    Paragraph("Northwind Labs - Quarterly Report Q3", styles["Title"]),
    Paragraph("Financial Summary", H),
    Paragraph("Revenue for the third quarter reached 4.2 million USD, an increase of 12% compared with the previous quarter. "
              "Growth was driven mainly by the launch of the Orbit Home Hub in the European market. Customer churn fell "
              "to 1.9%, the lowest level in two years.", B),
    Spacer(1, 10), table, Spacer(1, 10),
    PageBreak(),
    Paragraph("Outlook", H),
    Paragraph("Management expects Q4 revenue growth to remain in the high single digits. Two new engineering teams will be "
              "formed in Berlin and Lisbon. The planned headcount at the end of the year is 95.", B),
    Paragraph("Risks", H),
    Paragraph("The main risks identified for the coming quarter are component supply delays for the Orbit Home Hub, currency "
              "exposure on euro-denominated sales, and the time needed to hire senior engineers. Supply delays are expected "
              "to add roughly two weeks to delivery times in December.", B),
    Paragraph("Regional Breakdown", H),
    Paragraph("Europe contributed 61% of Q3 revenue, North America 29% and the rest of the world 10%. The Orbit Home Hub "
              "accounted for 38% of total revenue in its first full quarter on sale.", B),
])

# Image-only PDF: draws shapes but contains NO text layer (simulates a scan). Used to test the "needs OCR" path.
from reportlab.pdfgen import canvas

c = canvas.Canvas(str(OUT.parent / "edge_cases" / "scanned_like.pdf"), pagesize=A4)
c.rect(100, 500, 300, 200, fill=1)
c.save()
print("PDFs written to", OUT)
