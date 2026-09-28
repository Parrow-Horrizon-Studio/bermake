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
Bermake needs OpenGL 3.3. If your computer's graphics driver cannot provide
it, which is common in virtual machines and over Remote Desktop, Bermake
explains the problem and offers to restart using compatibility rendering.
Compatibility rendering draws through a bundled copy of Mesa instead of your
graphics driver. It works on almost any computer but may be slower.

You can turn it on or off at any time from Help > Use Compatibility
Rendering. To start once with it on, run:

    Bermake.exe --compatibility-rendering

Compatibility rendering uses Mesa's CPU renderer, llvmpipe, which needs no
graphics card. Advanced users can choose another Mesa driver by setting the
GALLIUM_DRIVER environment variable before starting Bermake.

Reporting a problem
-------------------
1. Open Help > About Bermake and click "Copy details".
2. Open an issue at https://github.com/Parrow-Horrizon-Studio/bermake/issues
   and paste the details.
3. Attach the log files. The About window shows their folder; it is usually
   %LOCALAPPDATA%\Parrow Horrizon Studio\Bermake\logs

If Bermake will not start at all, run this in a Command Prompt from this
folder and attach the report.json it writes:

    Bermake.exe --smoke-test report.json

Licences
--------
Bermake is free software under the GNU General Public License v3.0 or later;
see LICENSE.txt. Source code: https://github.com/Parrow-Horrizon-Studio/bermake
Bundled components and their licences are listed in THIRD-PARTY-NOTICES.txt.
