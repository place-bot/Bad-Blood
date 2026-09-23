# Computer-to-Raspberry-Pi deployment handoff

Copy all of [PROMPT_ZH.md](PROMPT_ZH.md) into Codex on the computer connected to
the microSD reader. It is a self-contained implementation task, not an installer.
The prompt is in Chinese for the operator; the requested dashboard is in English.

Goal: run the existing V1 model **on the Raspberry Pi**, with SBP/DBP and waveforms
visible in the computer's browser. No sensor is needed for the first phase:
use the repository's recorded PulseDB waveforms, labeled as **RECORDED DATA**.

At the time of this handoff, model inference exists but the Pi dashboard and
system service still need to be implemented and tested on the destination Pi.
No Pi benchmark or hardware deployment is claimed by these instructions.

The other Codex should stop only for actual physical actions, credentials,
disk-erasure confirmation or missing hardware. It should never infer that a
microSD reader can run the Pi model on behalf of the Pi.

## Physical sequence

1. Insert the microSD in the computer's reader. Identify the exact removable
   disk; preserve an existing course image unless replacing it is approved.
2. If needed, use official Raspberry Pi Imager to install a suitable 64-bit OS
   and configure the user, network and SSH. Writing an image erases the card.
3. Wait for write verification and safely eject the card in the OS.
4. Disconnect Pi power. Remove the card from the reader/SD adapter and insert
   the **microSD itself** into the Pi's microSD slot. Follow the slot orientation
   for the actual Pi model; do not force it.
5. Connect networking as configured and the correct Pi power supply. Boot,
   then let Codex connect through SSH from the computer.
6. Install the application on the running Pi. Open the display URL that Codex
   verifies. The computer is a display, not the inference host.
7. Shut down the Pi through its OS before removing power or the card.

## Sources

- [Raspberry Pi setup and imaging](https://www.raspberrypi.com/documentation/computers/getting-started.html)
- [Raspberry Pi SSH and remote access](https://www.raspberrypi.com/documentation/computers/remote-access.html)
- [Official prompting guidance](https://developers.openai.com/api/docs/guides/prompt-engineering)

Repository paths and model contract were checked against V1 source on 2026-09-23.
