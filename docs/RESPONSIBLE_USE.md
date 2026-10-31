# Responsible use

Face transfer changes identity-bearing visual information. A technically successful frame does not establish consent, authenticity, ownership or permission to present the output to another person.

## Intended scope

This public snapshot is designed for:

- experiments using your own face or a participant who gave informed permission;
- clearly disclosed demonstrations of computer-vision compositing;
- synthetic or licensed evaluation material;
- accessibility and avatar research where processed output is visible to participants;
- learning about frame transport, alignment, masking and restoration.

## Out of scope

Do not use this project to:

- impersonate another person;
- misrepresent your identity, presence, qualifications or actions;
- deceive an employer, interviewer, examiner, customer or identity-verification system;
- bypass liveness, anti-fraud, platform or workplace safeguards;
- process a person's image without an appropriate legal basis and permission;
- publish biometric source material, embeddings or output without authorization.

The historical project name is retained for repository continuity, not as approval for deceptive interview use.

## Public-code safeguards

The publication makes several boundaries concrete:

- one source image must be supplied explicitly for every server run;
- personalized weights, training recordings, reference galleries and facial embeddings are excluded;
- network services bind to loopback by default;
- the browser extension is limited to localhost and 127.0.0.1;
- the extension is receive-only and displays a permanent **DIGITALLY PROCESSED OUTPUT** label;
- the extension does not intercept media APIs, enumerate devices, alter device identity or conceal JavaScript modifications;
- no third-party conferencing or interview-service setup is documented.

These measures reduce accidental misuse; they do not replace the operator's responsibility.

## Data handling checklist

Before a consented experiment:

1. Record who supplied the source and target media and what use they accepted.
2. Keep source images and generated output outside the repository.
3. Use loopback networking or an authenticated tunnel.
4. Decide a deletion date for captured frames, logs and output.
5. Avoid collecting audio unless the experiment genuinely needs it.
6. Clearly label any demonstration and explain that the video is processed.
7. Stop immediately if a participant withdraws consent.

## Reporting concerns

Do not open a public issue containing a person's face, private recording, biometric template, credential or infrastructure address. Describe the software problem with synthetic data or a minimal, non-identifying reproduction.

