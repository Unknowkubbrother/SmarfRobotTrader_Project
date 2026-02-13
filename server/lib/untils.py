from random import randint
import os
import resend

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def send_email(to_email: str, subject: str, html_content: str):
    resend.api_key = os.getenv("RESEND_API_KEY")
    params = {
        "from": os.getenv("EMAIL_FROM", "SmarfRobotTrade <onboarding@resend.dev>"),
        "to": [to_email],
        "subject": subject,
        "html": html_content
    }
    email = resend.Emails.send(params)
    return email


def generate_otp_email_template(otp: str, purpose: str = "login") -> str:
    purpose_config = {
        "login": {
            "title": "Verification code",
            "message": "Enter the following verification code when prompted:",
        },
        "register": {
            "title": "Verification code",
            "message": "Enter the following verification code to complete your registration:",
        },
        "forgot_password": {
            "title": "Verification code",
            "message": "Enter the following verification code to reset your password:",
        },
    }

    config = purpose_config.get(purpose, purpose_config["login"])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config["title"]} - SmarfRobotTrade</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f4f4f5; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f4f4f5;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width: 560px;">
                    
                    <!-- Logo -->
                    <tr>
                        <td align="center" style="padding: 0 0 32px 0;">
                            <div style="width: 72px; height: 72px; background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); border-radius: 18px; display: inline-block; text-align: center; line-height: 72px; box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);">
                                <span style="font-size: 28px; font-weight: 800; color: #ffffff; font-family: 'SF Pro Display', -apple-system, sans-serif; letter-spacing: -1px;">SR</span>
                            </div>
                        </td>
                    </tr>

                    <!-- Main Card -->
                    <tr>
                        <td style="background-color: #ffffff; border-radius: 16px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);">
                            <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                                
                                <!-- Title & Message -->
                                <tr>
                                    <td style="padding: 48px 48px 0 48px;">
                                        <h1 style="margin: 0 0 12px 0; font-size: 28px; font-weight: 700; color: #09090b; letter-spacing: -0.5px;">{config["title"]}</h1>
                                        <p style="margin: 0; font-size: 16px; line-height: 1.6; color: #71717a;">{config["message"]}</p>
                                    </td>
                                </tr>

                                <!-- OTP Code -->
                                <tr>
                                    <td style="padding: 32px 48px;">
                                        <div style="font-size: 42px; font-weight: 800; color: #09090b; letter-spacing: 6px; font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;">{otp}</div>
                                    </td>
                                </tr>

                                <!-- Security Notice -->
                                <tr>
                                    <td style="padding: 0 48px 48px 48px;">
                                        <p style="margin: 0; font-size: 15px; color: #71717a; line-height: 1.5;">To protect your account, do not share this code.</p>
                                    </td>
                                </tr>

                            </table>
                        </td>
                    </tr>

                    <!-- Didn't Request Section -->
                    <tr>
                        <td style="padding: 32px 48px 0 48px;">
                            <p style="margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #09090b;">Didn&#39;t request this?</p>
                            <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #a1a1aa;">If you didn&#39;t make this request, you can safely ignore this email. This code will expire in <strong style="color: #71717a;">5 minutes</strong>.</p>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 32px 48px; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #d4d4d8;">&copy; 2026 SmarfRobotTrade</p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>'''

    return html


def send_otp_email(to_email: str, otp: str, purpose: str = "login"):
    subject_map = {
        "login": "Your Login Verification Code - SmarfRobotTrade",
        "register": "Complete Your Registration - SmarfRobotTrade",
        "forgot_password": "Password Reset Code - SmarfRobotTrade",
    }
    subject = subject_map.get(purpose, subject_map["login"])
    html_content = generate_otp_email_template(otp, purpose)
    return send_email(to_email, subject, html_content)
