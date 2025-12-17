# 처리
import os
import psutil
import webbrowser
import urllib.parse

programs = {"구글" : r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Google Chrome.lnk",
            "카카오톡" : r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\카카오톡.lnk",
            "워드": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Word.lnk",
            "엑셀": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Excel.lnk",
            "파워포인트": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\PowerPoint.lnk",
            "디스토드": r"C:\Users\showm\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Discord Inc\Discord.lnk",
            "스팀": r"C:\Users\showm\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Steam\Steam.lnk" ,
            "메모장": r"C:\Windows\System32\notepad.exe",
            }


program_name = {"구글" : 'chrome.exe',
                '카카오톡' : 'KaKaoTalk.exe',
                "워드": "WINWORD.EXE",
                "엑셀": "EXCEL.EXE",
                "파워포인트": "POWERPNT.EXE",
                "디스코드": "Discord.exe",
                "스팀": "Steam.exe",
                "메모장": "notepad.exe"
                }



def execute(text):
    text = (text.replace(' ', '')
    .replace('!', '')
    .replace('?', '')
    .replace('.', '')
    .replace(',', ''))
    if ("켜" in text) or ("혀" in text):
        action = 'run'
    elif ("꺼" in text) or ("닫아" in text) or ("다다" in text):
        action = "close"
    elif "검색" in text:
        action = "search"
    else:
        return

    program =  (text.replace('켜', "")
                    .replace('꺼', "")
                    .replace('혀','')
                    .replace('닫아','')
                    .replace('다다','')
                    .replace('검색',''))

    if action == 'run':
        for ko, file in programs.items():
            if ko == program:
                os.startfile(file)

    if action == 'close':
        for ko, file in program_name.items():
            if ko == program:
                for prog in psutil.process_iter(['pid', 'name']):
                    try:
                        if prog.info['name'] == file:
                            prog.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

    if action == 'search':
        a = urllib.parse.quote(program)
        url = f"https://www.google.com/search?q={program}"
        webbrowser.open(url)


