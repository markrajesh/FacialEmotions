# Test Image Fixtures

## Required Files

Place the following test images in this directory before running tests:

| Filename | Description | Expected Outcome |
|----------|-------------|-----------------|
| `single_face.jpg` | One person showing a clear happiness expression | happiness, genuine |
| `single_face_sad.jpg` | One person showing sadness | sadness |
| `neutral_face.jpg` | One person with neutral expression | neutral |
| `no_face.jpg` | Scene with no faces | 0 faces detected |

### Image Requirements

- Format: JPEG, PNG, or BMP
- Size: 224×224 px minimum, 1920×1080 maximum
- Consented subjects: All images must use photos where the subject has consented to use in machine learning research (e.g. public dataset samples, Creative Commons licensed photos, or photos of yourself)

### Provenance

Images sourced from the FER-2013 dataset subset or equivalent ethically-cleared public repository.
See [quickstart.md](../../specs/001-you-please-create/quickstart.md) for dataset ethics documentation.
