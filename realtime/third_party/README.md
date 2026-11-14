miniaudio.h is vendored, unmodified, from miniaudio tag 0.11.21:
https://github.com/mackron/miniaudio/blob/0.11.21/miniaudio.h

The original public-domain/MIT-0 dual-license notice is included at the end of
that header. It supplies ALSA device I/O and startup WAV decoding; processBlock
never calls the decoder or backend I/O. Backend hooks depend on this pinned
version and must be reviewed when upgrading miniaudio.
