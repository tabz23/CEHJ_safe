#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/*
 * SAPIEN 3.0.0b1 parsePCIString() only accepts 12-char (`0000:bb:dd.f`) or
 * 7-char (`bb:dd.f`) PCI ids. CUDA 12 cudaDeviceGetPCIBusId() writes
 * 16-char ids (`00000000:bb:dd.f`), which throws RuntimeError: invalid PCI
 * string from SapienRenderer().
 *
 * Two hooks:
 *   1. cudaDeviceGetPCIBusId — shrink the C string before SAPIEN strlen()s it
 *   2. parsePCIString — shrink a GCC COW std::string (jmp stub, original ABI)
 */

#define PATCH_LEN 12

static const unsigned char kParsePrologue[PATCH_LEN] = {
    0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x55, 0x48, 0x89, 0xfd};

static const unsigned char kCudaPrologue[PATCH_LEN] = {
    0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x41, 0x89, 0xd5, 0x55};

static int looks_like_16char_pci(const char *s) {
  int i;
  if (s == NULL || s[4] == ':' || s[8] != ':' || s[11] != ':' || s[14] != '.') {
    return 0;
  }
  for (i = 0; i < 8; i++) {
    char c = s[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
          (c >= 'A' && c <= 'F'))) {
      return 0;
    }
  }
  return 1;
}

static void shrink_cstr(char *s) {
  unsigned domain;
  char fixed[16];
  if (!looks_like_16char_pci(s)) {
    return;
  }
  domain = (unsigned)strtoul(s, NULL, 16);
  snprintf(fixed, sizeof(fixed), "%04x%s", domain, s + 8);
  fprintf(stderr, "[sapien_pci_hook] PCI '%s' -> '%s'\n", s, fixed);
  fflush(stderr);
  memcpy(s, fixed, 13);
}

static void shrink_pci_string(void *str) {
  char *p;
  size_t cow_len;
  static int logs = 0;
  if (str == NULL) {
    return;
  }
  p = *(char **)str;
  if (p == NULL) {
    return;
  }
  cow_len = *(size_t *)(p - 24);
  if (logs < 8) {
    logs++;
    fprintf(stderr, "[sapien_pci_hook] parsePCIString cow_len=%zu s='%.20s'\n",
            cow_len, p);
    fflush(stderr);
  }
  if (!looks_like_16char_pci(p)) {
    return;
  }
  shrink_cstr(p);
  if (cow_len == 16) {
    *(size_t *)(p - 24) = 12;
  }
}

static int mprotect_range(void *addr, size_t len, int prot) {
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  uintptr_t start = (uintptr_t)addr & ~(page - 1);
  uintptr_t end = ((uintptr_t)addr + len + page - 1) & ~(page - 1);
  return mprotect((void *)start, end - start, prot);
}

static unsigned char *emit_jmp(unsigned char *dst, void *to) {
  dst[0] = 0x48;
  dst[1] = 0xb8;
  memcpy(dst + 2, &to, 8);
  dst[10] = 0xff;
  dst[11] = 0xe0;
  return dst + 12;
}

static unsigned char *emit_call(unsigned char *dst, void *to) {
  dst[0] = 0x48;
  dst[1] = 0xb8;
  memcpy(dst + 2, &to, 8);
  dst[10] = 0xff;
  dst[11] = 0xd0;
  return dst + 12;
}

static unsigned char *make_rwx(size_t n) {
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return NULL;
  }
  return (unsigned char *)p;
}

static int patch_jmp(void *target, void *to) {
  unsigned char *p = (unsigned char *)target;
  if (mprotect_range(p, PATCH_LEN, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
    return -4;
  }
  emit_jmp(p, to);
  __builtin___clear_cache((char *)p, (char *)p + PATCH_LEN);
  return 0;
}

int install_parse_pci_hook(void *target) {
  unsigned char *p;
  unsigned char *stub;
  unsigned char *w;
  int rc;

  if (target == NULL) {
    return -1;
  }
  p = (unsigned char *)target;
  if (p[0] == 0x48 && p[1] == 0xb8) {
    return 0;
  }
  if (memcmp(p, kParsePrologue, PATCH_LEN) != 0) {
    fprintf(stderr, "[sapien_pci_hook] unexpected parsePCIString prologue at %p\n",
            target);
    fflush(stderr);
    return -5;
  }
  stub = make_rwx(64);
  if (stub == NULL) {
    return -3;
  }
  w = stub;
  *w++ = 0x57;
  w = emit_call(w, (void *)shrink_pci_string);
  *w++ = 0x5f;
  memcpy(w, p, PATCH_LEN);
  w += PATCH_LEN;
  emit_jmp(w, p + PATCH_LEN);
  __builtin___clear_cache((char *)stub, (char *)stub + 64);
  rc = patch_jmp(p, stub);
  if (rc != 0) {
    return rc;
  }
  fprintf(stderr, "[sapien_pci_hook] patched parsePCIString at %p\n", target);
  fflush(stderr);
  return 0;
}

typedef int (*cuda_pci_fn)(char *, int, int);

static cuda_pci_fn orig_cuda_pci = NULL;

static int hook_cuda_pci(char *buf, int len, int device) {
  int rc = orig_cuda_pci(buf, len, device);
  if (rc == 0 && buf != NULL) {
    fprintf(stderr, "[sapien_pci_hook] cudaDeviceGetPCIBusId rc=0 buf='%s'\n",
            buf);
    fflush(stderr);
    shrink_cstr(buf);
  } else {
    fprintf(stderr,
            "[sapien_pci_hook] cudaDeviceGetPCIBusId rc=%d device=%d len=%d\n",
            rc, device, len);
    fflush(stderr);
  }
  return rc;
}

int install_cuda_pci_hook(void *target) {
  unsigned char *p;
  unsigned char *tramp;
  int rc;

  if (target == NULL) {
    return -1;
  }
  p = (unsigned char *)target;
  if (p[0] == 0x48 && p[1] == 0xb8) {
    return 0;
  }
  if (memcmp(p, kCudaPrologue, PATCH_LEN) != 0) {
    fprintf(stderr,
            "[sapien_pci_hook] unexpected cudaDeviceGetPCIBusId prologue at %p\n",
            target);
    fflush(stderr);
    return -5;
  }
  tramp = make_rwx(64);
  if (tramp == NULL) {
    return -3;
  }
  memcpy(tramp, p, PATCH_LEN);
  emit_jmp(tramp + PATCH_LEN, p + PATCH_LEN);
  __builtin___clear_cache((char *)tramp, (char *)tramp + 64);
  orig_cuda_pci = (cuda_pci_fn)tramp;
  rc = patch_jmp(p, (void *)hook_cuda_pci);
  if (rc != 0) {
    return rc;
  }
  fprintf(stderr, "[sapien_pci_hook] patched cudaDeviceGetPCIBusId at %p\n",
          target);
  fflush(stderr);
  return 0;
}
