#!/usr/bin/env python3
"""
简易训练日志浏览工具 - 实时查看训练进度、损失和时间
"""
import os
import sys
import re
from pathlib import Path
from collections import deque


def parse_log_line(line):
    """提取日志中的关键信息"""
    # 支持多种日志格式
    
    # 格式1: [Iter N] flow: X; d: Y; hole: Z; valid: W
    iter_match = re.search(r'\[Iter (\d+)\]', line)
    flow_match = re.search(r'flow:\s*([0-9.]+)', line)
    d_match = re.search(r'd:\s*([0-9.]+)', line)
    hole_match = re.search(r'hole:\s*([0-9.]+)', line)
    valid_match = re.search(r'valid:\s*([0-9.]+)', line)
    
    # 格式2: 旧格式 [N/M] Loss_gen Loss_dis
    if not iter_match:
        iter_match = re.search(r'\[(\d+)/(\d+)\]', line)
        loss_gen_match = re.search(r'Loss_gen[:\s=]+([0-9.]+)', line, re.IGNORECASE)
        loss_dis_match = re.search(r'Loss_dis[:\s=]+([0-9.]+)', line, re.IGNORECASE)
    else:
        loss_gen_match = None
        loss_dis_match = None
    
    # 查找时间信息 (支持多种格式)
    time_match = re.search(r'(\d{2}):(\d{2}):(\d{2})', line)
    
    return {
        'iter': int(iter_match.group(1)) if iter_match and isinstance(iter_match.group(1), str) else None,
        'flow': float(flow_match.group(1)) if flow_match else None,
        'd': float(d_match.group(1)) if d_match else None,
        'hole': float(hole_match.group(1)) if hole_match else None,
        'valid': float(valid_match.group(1)) if valid_match else None,
        'loss_gen': float(loss_gen_match.group(1)) if loss_gen_match else None,
        'loss_dis': float(loss_dis_match.group(1)) if loss_dis_match else None,
        'time': time_match.group(0) if time_match else None,
        'raw': line
    }


def view_log(log_file, tail_lines=50, watch=False):
    """
    查看训练日志
    
    Args:
        log_file: 日志文件路径
        tail_lines: 显示最后 N 行
        watch: 是否实时监视（类似 tail -f）
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"📋 日志文件: {log_file}")
    print(f"📊 显示最后 {tail_lines} 行")
    print("-" * 100)
    
    last_pos = 0
    iteration_history = deque(maxlen=10)  # 保留最后 10 次迭代的信息
    
    try:
        while True:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 如果文件已读过，从上次位置继续
                f.seek(last_pos)
                new_lines = f.readlines()
                last_pos = f.tell()
            
            if new_lines:
                # 只显示包含有意义内容的行
                for line in new_lines:
                    line = line.rstrip()
                    
                    # 过滤日志（只显示关键信息和错误）
                    if '[Iter' in line or any(keyword in line for keyword in [
                        'Epoch', 'WARNING', 'ERROR', 'Saved', 'checkpoint', 'create folder', '[**]'
                    ]):
                        parsed = parse_log_line(line)
                        
                        # 如果有迭代信息，保存到历史记录
                        if parsed['iter']:
                            iteration_history.append(parsed)
                            print(f"✓ [Iter {parsed['iter']}]", end="")
                            if parsed['flow']:
                                print(f" flow: {parsed['flow']:.4f}", end="")
                            if parsed['d']:
                                print(f" d: {parsed['d']:.4f}", end="")
                            if parsed['hole']:
                                print(f" hole: {parsed['hole']:.4f}", end="")
                            if parsed['valid']:
                                print(f" valid: {parsed['valid']:.4f}", end="")
                            if parsed['time']:
                                print(f" @{parsed['time']}", end="")
                            print()
                        else:
                            # 打印其他关键行
                            if 'tensorflow' not in line.lower() and 'onednn' not in line.lower():
                                print(f"ℹ️ {line[:100]}")
                
                # 实时显示统计
                if iteration_history:
                    last_iter = iteration_history[-1]
                    if last_iter['iter']:
                        print(f"\n📈 最新: 第 {last_iter['iter']} 步", end="")
                        if last_iter['flow']:
                            print(f" | flow: {last_iter['flow']:.4f}", end="")
                        if last_iter['d']:
                            print(f" | d: {last_iter['d']:.4f}", end="")
                        if last_iter['hole']:
                            print(f" | hole: {last_iter['hole']:.4f}", end="")
                        if last_iter['valid']:
                            print(f" | valid: {last_iter['valid']:.4f}", end="")
                        print()
                
                if not watch:
                    break
                    
            else:
                if not watch:
                    break
                # 等待新内容
                import time
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\n⏹️ 已停止监视")
        return
    except Exception as e:
        print(f"❌ 错误: {e}")
        return


def show_summary(log_file):
    """显示日志的摘要统计"""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"\n📊 日志摘要统计")
    print("=" * 100)
    
    iterations = []
    losses_flow = []
    losses_d = []
    losses_hole = []
    losses_valid = []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed['iter']:
                iterations.append(parsed['iter'])
            if parsed['flow']:
                losses_flow.append(parsed['flow'])
            if parsed['d']:
                losses_d.append(parsed['d'])
            if parsed['hole']:
                losses_hole.append(parsed['hole'])
            if parsed['valid']:
                losses_valid.append(parsed['valid'])
    
    if iterations:
        print(f"✓ 总迭代数: {len(iterations)}")
        print(f"  进度: [{iterations[-1]}/100]")
    
    if losses_flow:
        print(f"\n流估计损失 (Flow Loss):")
        print(f"  最新值: {losses_flow[-1]:.6f}")
        print(f"  平均值: {sum(losses_flow)/len(losses_flow):.6f}")
        print(f"  最小值: {min(losses_flow):.6f}")
        print(f"  最大值: {max(losses_flow):.6f}")
    
    if losses_d:
        print(f"\n判别器损失 (Discriminator Loss):")
        print(f"  最新值: {losses_d[-1]:.6f}")
        print(f"  平均值: {sum(losses_d)/len(losses_d):.6f}")
        print(f"  最小值: {min(losses_d):.6f}")
        print(f"  最大值: {max(losses_d):.6f}")
    
    if losses_hole:
        print(f"\n空洞损失 (Hole Loss):")
        print(f"  最新值: {losses_hole[-1]:.6f}")
        print(f"  平均值: {sum(losses_hole)/len(losses_hole):.6f}")
        print(f"  最小值: {min(losses_hole):.6f}")
        print(f"  最大值: {max(losses_hole):.6f}")
    
    if losses_valid:
        print(f"\n有效区域损失 (Valid Loss):")
        print(f"  最新值: {losses_valid[-1]:.6f}")
        print(f"  平均值: {sum(losses_valid)/len(losses_valid):.6f}")
        print(f"  最小值: {min(losses_valid):.6f}")
        print(f"  最大值: {max(losses_valid):.6f}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="简易训练日志浏览工具")
    parser.add_argument("log_file", nargs="?", default="training_small_v3.log",
                        help="日志文件路径 (默认: training_small_v3.log)")
    parser.add_argument("-n", "--lines", type=int, default=50,
                        help="显示最后 N 行 (默认: 50)")
    parser.add_argument("-w", "--watch", action="store_true",
                        help="实时监视日志 (按 Ctrl+C 停止)")
    parser.add_argument("-s", "--summary", action="store_true",
                        help="显示摘要统计")
    
    args = parser.parse_args()
    
    print("🚀 E2FGVI 训练日志浏览工具")
    print()
    
    if args.summary:
        show_summary(args.log_file)
    else:
        view_log(args.log_file, tail_lines=args.lines, watch=args.watch)
