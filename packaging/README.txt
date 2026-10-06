Bermake for Windows
===================

Thank you for testing Bermake. This is an early build for testers: it is not
signed and it has no installer. To remove it, delete this folder.

Starting Bermake
----------------
Double-click Bermake.exe.

Windows may show "Windows protected your PC". That is because this build is
not code-signed yet. Click "More info", then "Run anyway".

If the 3D view does not work
----------------------------
Bermake needs OpenGL 3.3. Every time it starts, before the main window
opens, it checks what your computer's graphics driver provides. If the
driver cannot provide OpenGL 3.3, or provides no OpenGL at all, which is
common in virtual machines, in Windows Sandbox and over Remote Desktop,
Bermake explains the problem and offers to restart using compatibility
rendering. The check also draws a small test image, so a driver that reports
OpenGL but draws nothing gets the same message and offer. Compatibility
rendering draws through a bundled copy of Mesa instead of your graphics
driver. It works on almost any computer but may be slower.

You can turn it on or off at any time from Help > Use Compatibility
Rendering. To start once with it on, run:

    Bermake.exe --compatibility-rendering

If you decline the offer to restart, or none is offered, Bermake still opens, but the 3D view
stays empty apart from a short note saying that Bermake cannot draw it on this
computer. The note says to turn on Help > Use Compatibility Rendering and
restart when that is possible, and otherwise points to Help > About Bermake
for the details to include in a bug report.

Compatibility rendering uses Mesa's CPU renderer, llvmpipe, which needs no
graphics card. It is the only supported setting. Do not set the
GALLIUM_DRIVER environment variable: the d3d12 driver is not supported in this
build and closes Bermake without a message.

If Bermake will not start after you turned compatibility rendering on, start
it once with it off by running this in a Command Prompt from this folder:

    Bermake.exe --no-compatibility-rendering

Then turn it off for good from Help > Use Compatibility Rendering.

Autosave
--------
Bermake autosaves your unsaved work every 5 minutes by default. You can
change the interval, or turn autosave off, in File > Autosave. It saves when
you pause, not in the middle of a drag, and your own file is never touched.
After each autosave the status bar shows "Autosaved" and the time, until you
make your next change.

If Bermake closes unexpectedly, the next launch offers to recover the work.
Recovered work opens as unsaved, with "(recovered)" in the title; Save asks
where to put it, starting at the original file's folder and name, if it had one.

Autosaves are kept in this folder:

    %LOCALAPPDATA%\Parrow Horrizon Studio\Bermake\recovery

If a recovery file cannot be opened, Bermake keeps it there as
something.broken.berm instead of deleting it. Attach that file to your bug
report.

Reporting a problem
-------------------
1. Open Help > About Bermake and click "Copy details".
2. Open an issue at https://github.com/Parrow-Horrizon-Studio/bermake/issues
   and paste the details.
3. Attach the log files. The About window shows their folder; it is usually
   %LOCALAPPDATA%\Parrow Horrizon Studio\Bermake\logs

If Bermake will not start at all, run this in a Command Prompt from this
folder and attach the report.json it writes. "start /wait" makes the Command
Prompt wait until the test has finished:

    start /wait Bermake.exe --smoke-test report.json

Licences
--------
Bermake is free software under the GNU General Public License v3.0 or later;
see LICENSE.txt. Source code: https://github.com/Parrow-Horrizon-Studio/bermake
Bundled components and their licences are listed in THIRD-PARTY-NOTICES.txt.
