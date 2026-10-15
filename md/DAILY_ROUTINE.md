# Ежедневная работа с GPU-компьютером

| | |
|---|---|
| Адрес домашнего ПК (Radmin VPN) | `26.244.190.10` |
| Пользователь | `kirill` |
| Проект на домашнем ПК | `C:\Users\kirill\Desktop\TATPN_diplom` |

## Остановить и перезапустить с обновлённым кодом

Все команды - в PowerShell на домашнем ПК, в папке проекта (после входа по ssh
сначала `powershell`, затем `cd C:\Users\kirill\Desktop\TATPN_diplom`).

**1. Остановить цикл вместе с этапом** (нужно убить дерево процессов, иначе
`run_all` запустит следующий этап):

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_all\.py|run_stage\d\.py' } | ForEach-Object { taskkill /F /T /PID $_.ProcessId }
```

Проверка (вывод должен быть пустым; если процесс виден - повторить команду):

```powershell
Get-Process python -ErrorAction SilentlyContinue
```

**2. Получить исправления:**

```powershell
git pull
```

**3. Очистить старые результаты** (чтобы новые сводки не смешивались со старыми;
исходные данные и логи запусков не затрагиваются):

```powershell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue processed_data, logs\stage1_baselines, logs\stage2_cnn, logs\stage3_cnn_mamba, logs\stage4_target, logs\stage5_tatpn, logs\run_all, artifacts\stage2, artifacts\stage3, artifacts\stage4, artifacts\stage5, reports\stage6, run_all.out
```

**4. Первый прогон - по одному этапу с проверкой сводки** (этапы 0-2 занимают
секунды-минуты). После каждого смотреть сводку и идти дальше, только если нет
`FAIL`:

```powershell
.venv\Scripts\python.exe run_stage0.py
```

```powershell
.venv\Scripts\python.exe run_stage1.py
Get-Content logs\stage1_baselines\summary.txt -Encoding UTF8
```

```powershell
.venv\Scripts\python.exe run_stage2.py
Get-Content logs\stage2_cnn\summary.txt -Encoding UTF8
```

**5. Замерить скорость на исправленном скане** (несколько минут, покажет время
эпохи и подбор батча; по нему видно, сколько займут этапы 3-5):

```powershell
.venv\Scripts\python.exe benchmark_gpu.py
```

**6. Этапы 3-5 и 6 - в фоне, одной командой** (с этапа 3, с остановкой на первом
этапе с непройденными проверками):

```powershell
$P = (Get-Location).Path
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"cd /d $P && .venv\Scripts\python.exe -u run_all.py --from 3 --stop-on-fail > run_all.out 2>&1`"" }
```

Нужно `ReturnValue : 0`. После этого окно ssh можно закрыть.

**Весь цикл с нуля одной командой** (когда этапы 0-2 уже проверены и доверие
есть) - так же, но без `--from 3 --stop-on-fail`: `run_all.py`.

### Следить без повторных запросов

Живой поток журнала (строки приходят по мере выполнения, выход - `Ctrl+C`,
цикл не затрагивается):

```powershell
Get-Content run_all.out -Wait -Tail 20 -Encoding UTF8
```

Сводка с автообновлением каждые 30 секунд до завершения цикла:

```powershell
.venv\Scripts\python.exe status.py --watch 30
```

С рабочего ПК без входа в сессию - разовый снимок (`--watch` по ssh без
терминала не нужен):

```text
ssh kirill@26.244.190.10 "cd /d C:\Users\kirill\Desktop\TATPN_diplom && .venv\Scripts\python.exe status.py"
```

### Частые вопросы

**По одному этапу или целиком?** Первый прогон после изменений кода - по одному
(пункт 4): ошибка видна сразу и на нужном этапе, а этапы 0-2 дешёвые. Целиком
(`run_all.py`) - когда всё проверено и нужен долгий прогон без присмотра.
`run_all.py --from N` продолжает с этапа N, не повторяя предыдущие.

**Не плодит ли повторный запуск новые данные?** Нет. Этап 0 детерминирован
(`RANDOM_SEED`): два прогона дают побитово одинаковый датасет, и он
перезаписывает тот же файл `processed_data/all_trajectories.npz`. Чекпоинты и
сводки тоже перезаписываются. Накапливаются только журналы запусков в
`logs/run_logs/` (по файлу на запуск этапа, небольшие) - их можно удалять.
`--from 1` и выше этап 0 не повторяет.

## 1. На рабочем ПК (Windows 10)

1. Запустить Radmin VPN, убедиться, что домашний ПК "в сети".
2. В PowerShell проверить порт (нужно `TcpTestSucceeded : True`):

   ```powershell
   Test-NetConnection 26.244.190.10 -Port 22
   ```

3. Подключиться (пароль учётной записи, не PIN):

   ```powershell
   ssh kirill@26.244.190.10
   ```

## 2. На домашнем ПК (внутри ssh-сессии)

4. Войти в PowerShell, перейти в проект, подтянуть код:

   ```
   powershell
   cd C:\Users\kirill\Desktop\TATPN_diplom
   .venv\Scripts\activate
   git pull
   ```

5. Посмотреть, что идёт и как нагружена видеокарта:

   ```powershell
   Get-Process python -ErrorAction SilentlyContinue
   nvidia-smi
   ```

6. Посмотреть журнал последнего запуска:

   ```powershell
   $log = Get-ChildItem logs\run_logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
   Get-Content $log.FullName -Tail 25
   ```

7. Запустить этап (`N` - номер этапа):

   - **этапы 0, 1, 2, 6** - прямо в сессии:

     ```powershell
     python run_stageN.py
     ```

   - **этапы 3, 4, 5** - в фоне, чтобы обрыв ssh не остановил обучение:

     ```powershell
     $P = (Get-Location).Path
     Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"cd /d $P && .venv\Scripts\python.exe run_stageN.py > stageN.out 2>&1`"" }
     ```

     Нужно `ReturnValue = 0`. После этого ssh можно закрыть.
     При первом запуске проверить: закрыть ssh, зайти снова, `Get-Process python` - процесс должен быть жив.

8. Прочитать результат:

   ```powershell
   Get-Content logs\stage5_tatpn\summary.txt
   ```

   Папки сводок: `stage1_baselines`, `stage2_cnn`, `stage3_cnn_mamba`, `stage4_target`, `stage5_tatpn`.

## 3. Снова на рабочем ПК

9. Забрать сводку (новое окно PowerShell, не внутри ssh):

   ```powershell
   scp kirill@26.244.190.10:C:/Users/kirill/Desktop/TATPN_diplom/logs/stage5_tatpn/summary.txt .
   ```

## Порядок этапов

Следующий этап - только после того, как предыдущий прошёл проверки в своей сводке.
`run_stage0.py` -> `1` -> `2` -> `3` -> `4` -> `5` -> `6`.

## Полный цикл одной командой

`run_all.py` запускает этапы 0-6 подряд, на всех эпохах, и замеряет реальное
время каждого. Если этап упал - цикл останавливается; если у этапа не прошли
проверки - идёт дальше и помечает это в итоге.

В фоне (в PowerShell, в папке проекта):

```powershell
$P = (Get-Location).Path
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"cd /d $P && .venv\Scripts\python.exe -u run_all.py > run_all.out 2>&1`"" }
```

Проверять состояние - одной командой, постоянно следить не нужно. Цикл идёт
сам, сам останавливается при ошибке и сам пишет итог.

С рабочего ПК, без входа на домашний (спросит пароль `kirill`):

```text
ssh kirill@26.244.190.10 "cd /d C:\Users\kirill\Desktop\TATPN_diplom && .venv\Scripts\python.exe status.py"
```

Или внутри ssh-сессии, в папке проекта:

```text
.venv\Scripts\python.exe status.py
```

Показывает: работает ли процесс, какой этап идёт, эпоху, завершённые этапы
с временем, загрузку видеокарты, а когда цикл закончился - итог. Если процесса
нет, а итога нет - пишет, что цикл остановился, и показывает последние
строки журнала.

Достаточно заглядывать на границах этапов, а не по минутам.

Продолжить с места остановки: `--from N` (например, `run_all.py --from 3`).

## Если не подключается

- Домашний ПК в сне или перезагрузился - включить, при необходимости войти в учётную запись на нём.
- Radmin VPN не запущен на одном из ПК.
- ПК "в сети", но порт закрыт - на домашнем ПК `Start-Service sshd`, проверить правило брандмауэра для порта 22.
- Рабочая сеть блокирует VPN.

## Остановить запуск

```powershell
Stop-Process -Name python -Force
```

Останавливает все процессы python на домашнем ПК.
