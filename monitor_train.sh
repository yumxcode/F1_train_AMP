#!/bin/bash
# F1 X1 训练监控脚本 v2
# 每 interval 秒检查一次任务状态。
# 检测到训练结束或异常则退出并提示。
#
# 用法:
#   前台运行: bash monitor_train.sh
#   后台运行: nohup bash monitor_train.sh > /tmp/train_monitor.out 2>&1 &
#   查看状态: cat .oma/experiments/exp-20260527-001/monitor.log | tail -20

TASK_ID="TASK_20260527_151"
INTERVAL=60  # 每60秒检查一次
LOG_FILE=".oma/experiments/exp-20260527-001/monitor.log"
PID_FILE=".oma/experiments/exp-20260527-001/monitor.pid"

mkdir -p "$(dirname "$LOG_FILE")"

echo "==========================================" | tee -a "$LOG_FILE"
echo "F1 X1 训练监控启动" | tee -a "$LOG_FILE"
echo "任务: $TASK_ID" | tee -a "$LOG_FILE"
echo "间隔: ${INTERVAL}s" | tee -a "$LOG_FILE"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

save_pid() {
    echo $$ > "$PID_FILE"
}

check_status() {
    local now=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 获取任务信息（通过tr和grep提取字段）
    local info_output=$(gm task info --task-id "$TASK_ID" 2>&1)
    local task_status=$(echo "$info_output" | tr ',' '\n' | grep '"taskStatus"' | grep -o '[0-9]' | head -1)
    local end_time=$(echo "$info_output" | tr ',' '\n' | grep '"endTime"' | grep -o '"[^"]*"' | tail -1)
    
    echo "[$now] status=$task_status endTime=$end_time" >> "$LOG_FILE"
    
    # 获取训练日志（最近10行）
    local log_lines=$(gm task logs --task-id "$TASK_ID" --raw --no-request-log 2>&1 | tail -10)
    if [ -n "$log_lines" ]; then
        echo "[$now] log_sample: $(echo "$log_lines" | tr '\n' ';' | head -c 200)" >> "$LOG_FILE"
        
        # 检查异常关键词
        if echo "$log_lines" | grep -qiE "error|exception|traceback|out of memory|nan|inf"; then
            echo "[$now] ⚠️ 检测到训练异常！" | tee -a "$LOG_FILE"
            echo "$log_lines" | tee -a "$LOG_FILE"
            return 1
        fi
        
        # 检查训练完成标志（最后一个save/iteration）
        if echo "$log_lines" | grep -q "Learning iteration"; then
            local iter=$(echo "$log_lines" | grep "Learning iteration" | grep -oP 'iteration \K\d+' | head -1)
            echo "[$now] 当前迭代: $iter" >> "$LOG_FILE"
        fi
    fi
    
    # 任务状态码: 0=草稿, 1=排队中, 2=初始化, 3=运行中, 4=已完成, 5=已终止, 6=失败
    case "$task_status" in
        4)
            echo "[$now] ✅ 训练已完成(status=4)" | tee -a "$LOG_FILE"
            return 0
            ;;
        5)
            echo "[$now] ⛔ 训练已终止(status=5)" | tee -a "$LOG_FILE"
            return 1
            ;;
        6)
            echo "[$now] ❌ 训练失败(status=6)" | tee -a "$LOG_FILE"
            return 1
            ;;
        "")
            echo "[$now] ⚠️ 无法获取状态" >> "$LOG_FILE"
            return 2
            ;;
    esac
    
    return 2  # 运行中
}

# 主循环
save_pid
while true; do
    check_status
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "✅ 正常结束，监控退出" | tee -a "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 0
    elif [ $rc -eq 1 ]; then
        echo "⚠️ 异常！监控退出" | tee -a "$LOG_FILE"
        echo "请查看: gm task logs --task-id $TASK_ID"
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep "$INTERVAL"
done
