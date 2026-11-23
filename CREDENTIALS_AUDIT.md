# Credentials & Security Review

**Project:** `3-kaggle/misconception-detection`  
**Review Date:** 2026-11-15  
**Audit Standard:** Safe Credentials Protocol (`/credentials`)  
**Status:** ⚠️ ACTION REQUIRED  

---

## 1. Environment & Config Files

- **Detected `.env*` Files:** `.env`
- **Tracked in Git:** None (Compliant)
- **Git Ignore Protection (`.gitignore`):** Configured (`.env` protected)

## 2. Source Code Hardcoded Secrets Scan

| File | Line | Type | Redacted Signature |
|---|---|---|---|
| `.venv-soup/lib/python3.15/site-packages/psutil/_pslinux.py` | 1138 | OpenAI API Key / Generic Secret | `sk-w...nux` |

> [!CAUTION]
> Hardcoded credential tokens or sensitive signatures were flagged in the files above. Move them to an untracked local `.env` or system environment variables.

## 3. Protocol Remediation & Best Practices

1. **Never Commit Secrets:** Ensure `.env` is listed in `.gitignore` and never committed into Git history.
2. **Safe Verification Protocol:** Check presence of required API keys using quiet mode without outputting values:
   ```bash
   grep -sq "^KEY_NAME=" ~/.env
   ```
3. **Adding Missing Keys:** Never paste secrets into chat or terminal history. Prompt via hidden input:
   ```bash
   printf "Enter KEY_NAME (typing hidden): " && read -s val && echo && echo "KEY_NAME=$val" >> ~/.env && echo "Saved."
   ```
4. **Environment Isolation:** Prefer loading credentials via standard libraries (`python-dotenv`, `dotenv`, or native process environment) rather than hardcoded configuration.
