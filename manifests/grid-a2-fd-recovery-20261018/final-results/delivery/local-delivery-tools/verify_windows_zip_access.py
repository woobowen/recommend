"""Read the completed ZIP through its actual Windows UNC path."""
from pathlib import Path
import datetime as dt
import json
import subprocess

BASE=Path(__file__).resolve().parent

def main():
    request=json.loads((BASE/'deliveries/package-build-request.json').read_text())
    path=Path(request['output'])
    verified=json.loads(path.with_suffix('.zip.verification.json').read_text())
    assert verified['status']=='PACKAGE_INTEGRITY_PASS'
    windows=subprocess.run(['wslpath','-w',str(path)],check=True,stdout=subprocess.PIPE,text=True).stdout.strip()
    assert windows.startswith('\\\\wsl.') or windows.startswith('\\\\wsl$\\'),windows
    literal="'"+windows.replace("'","''")+"'"
    script="$gridZip = Get-Item -LiteralPath "+literal+"; $gridZipHash = (Get-FileHash -LiteralPath "+literal+" -Algorithm SHA256).Hash; [ordered]@{path=$gridZip.FullName; bytes=$gridZip.Length; sha256=$gridZipHash} | ConvertTo-Json -Compress"
    process=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',script],
                           check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
    actual=json.loads(process.stdout)
    assert actual['bytes']==verified['bytes'] and actual['sha256'].lower()==verified['sha256']
    report=dict(status='WINDOWS_UNC_FILE_AND_SHA256_ACCESS_PASS',checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                WSL_path=str(path),Windows_UNC_path=windows,bytes=actual['bytes'],sha256=actual['sha256'].lower(),
                Windows_binary_file_read=True,second_ZIP_copy_created=False)
    destination=path.with_suffix('.zip.windows-access.json');assert not destination.exists()
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
