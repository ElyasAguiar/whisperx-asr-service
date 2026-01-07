# Security Considerations

## Dependency Vulnerabilities

### PyTorch 2.3.0 - Known Vulnerabilities

**Status**: Acknowledged - Risk Mitigated

The service uses PyTorch 2.3.0, which has known vulnerabilities related to `torch.load()`:
- CVE: PyTorch `torch.load` with `weights_only=True` leads to remote code execution
- Affected versions: < 2.6.0

**Why we continue to use 2.3.0**:
1. **CUDA Compatibility**: PyTorch 2.3.0 is specifically chosen for CUDA 12.1 compatibility with the base image
2. **Controlled Environment**: The service runs in a containerized environment with limited attack surface
3. **No User Model Uploads**: The service does NOT accept user-uploaded model files or pickle files

**Mitigation Strategy**:
- ✅ Service only loads models from trusted sources (Hugging Face, official repositories)
- ✅ No user-provided pickle files are loaded
- ✅ Docker container runs with minimal privileges
- ✅ Service does not expose model loading functionality to users
- ✅ All model files come from verified, trusted sources

**Attack Vector Analysis**:
The vulnerability requires an attacker to:
1. Upload a malicious pickle file to the server
2. Have the server load it using `torch.load()`

Our service:
- ❌ Does NOT accept user-uploaded model files
- ❌ Does NOT provide endpoints for loading arbitrary files
- ✅ Only loads pre-approved models from Hugging Face
- ✅ Uses environment variables to specify models (admin-controlled)

**Future Actions**:
When upgrading to a newer base image or PyTorch version, test compatibility and upgrade to PyTorch 2.6.0+ to fully resolve this vulnerability.

## Security Summary

| Vulnerability | Status | Risk Level | Mitigation |
|---------------|--------|------------|------------|
| FastAPI ReDoS (≤ 0.109.0) | ✅ Fixed | N/A | Upgraded to 0.115.6 |
| python-multipart DoS (< 0.0.18) | ✅ Fixed | N/A | Upgraded to 0.0.18 |
| python-multipart ReDoS (≤ 0.0.6) | ✅ Fixed | N/A | Upgraded to 0.0.18 |
| PyTorch torch.load RCE (< 2.6.0) | ⚠️ Acknowledged | Low | Service doesn't load user files |

**Overall Security Posture**: ✅ Acceptable for production use with documented risk acceptance.

---

Last Updated: 2026-01-07
