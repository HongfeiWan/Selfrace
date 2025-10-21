#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路网生成器和编辑器
支持可视化绘制、编辑路网，并保存为JSON格式
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import math
from typing import List, Tuple, Optional, Dict
import copy

class RoadSegment:
    """路段类，表示一个路段"""
    
    def __init__(self, points: List[Tuple[float, float]], road_id: int = 0, lane_id: int = -1, width: float = 4.0):
        """
        初始化路段
        
        Args:
            points: 路段的点列表 [(x, y), ...]
            road_id: 道路ID
            lane_id: 车道ID
            width: 车道宽度（米）
        """
        self.points = points
        self.road_id = road_id
        self.lane_id = lane_id
        self.width = width
        self.selected = False
        self.canvas_id = None  # Canvas上的绘制ID
    
    def to_quads(self, start_poly_id: int = 0) -> List[Dict]:
        """
        将路段转换为四边形列表
        Args:
            start_poly_id: 起始多边形ID
            
        Returns:
            四边形列表
        """
        quads = []
        poly_id = start_poly_id

        # 为每对相邻点生成四边形
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            # 计算垂直于路段方向的单位向量
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)
            
            if length < 0.001:  # 避免除零
                continue
            
            # 垂直向量
            perp_x = -dy / length * self.width / 2
            perp_y = dx / length * self.width / 2
            
            # 生成四边形的四个顶点
            vertices = [
                {"x": p1[0] - perp_x, "y": p1[1] - perp_y, "z": 0.0},
                {"x": p1[0] + perp_x, "y": p1[1] + perp_y, "z": 0.0},
                {"x": p2[0] + perp_x, "y": p2[1] + perp_y, "z": 0.0},
                {"x": p2[0] - perp_x, "y": p2[1] - perp_y, "z": 0.0}
            ]
            
            # 计算q值（沿路径的距离）
            q = sum(math.sqrt((self.points[j+1][0] - self.points[j][0])**2 + 
                             (self.points[j+1][1] - self.points[j][1])**2) 
                   for j in range(i))
            
            quad = {
                "polyId": poly_id,
                "vertices": vertices,
                "road_id": self.road_id,
                "lane_id": self.lane_id,
                "q": round(q, 2)
            }
            
            quads.append(quad)
            poly_id += 1
        
        return quads
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """获取路段的边界框"""
        if not self.points:
            return 0, 0, 0, 0
        
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        
        half_width = self.width / 2
        return (min(xs) - half_width, min(ys) - half_width,
                max(xs) + half_width, max(ys) + half_width)
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在路段附近"""
        # 简化版本：检查点到路段上任意线段的距离
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            dist = self._point_to_segment_distance(x, y, p1[0], p1[1], p2[0], p2[1])
            if dist <= self.width:
                return True
        
        return False
    
    @staticmethod
    def _point_to_segment_distance(px: float, py: float, 
                                   x1: float, y1: float, 
                                   x2: float, y2: float) -> float:
        """计算点到线段的距离"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


class RoadNetworkEditor:
    """路网编辑器主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("路网生成器和编辑器")
        self.root.geometry("1400x900")
        
        # 数据
        self.segments: List[RoadSegment] = []
        self.current_segment_points = []  # 当前正在绘制的路段的点
        self.selected_segment: Optional[RoadSegment] = None
        self.map_name = "CustomMap"
        
        # 绘图参数
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.scale = 1.0  # 缩放比例
        self.grid_size = 50  # 网格大小（像素）
        
        # 工具状态
        self.current_tool = "select"  # select, line, curve
        self.next_road_id = 0
        self.next_lane_id = -1
        self.lane_width = 4.0
        
        # 鼠标状态
        self.mouse_x = 0
        self.mouse_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        
        self._setup_ui()
        self._bind_events()
        
    def _setup_ui(self):
        """设置用户界面"""
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 左侧工具栏
        toolbar = ttk.Frame(main_container, width=250, relief=tk.RAISED, borderwidth=2)
        toolbar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 工具选择
        ttk.Label(toolbar, text="绘图工具", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.tool_var = tk.StringVar(value="select")
        
        tools = [
            ("选择工具", "select"),
            ("直线工具", "line"),
            ("曲线工具", "curve")
        ]
        
        for text, value in tools:
            rb = ttk.Radiobutton(toolbar, text=text, variable=self.tool_var, 
                                value=value, command=self._on_tool_change)
            rb.pack(anchor=tk.W, padx=10, pady=5)
        
        ttk.Separator(toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 参数设置
        ttk.Label(toolbar, text="路段参数", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Road ID
        param_frame1 = ttk.Frame(toolbar)
        param_frame1.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(param_frame1, text="Road ID:").pack(side=tk.LEFT)
        self.road_id_var = tk.IntVar(value=0)
        road_id_spinbox = ttk.Spinbox(param_frame1, from_=0, to=10000, 
                                      textvariable=self.road_id_var, width=10)
        road_id_spinbox.pack(side=tk.RIGHT)
        
        # Lane ID
        param_frame2 = ttk.Frame(toolbar)
        param_frame2.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(param_frame2, text="Lane ID:").pack(side=tk.LEFT)
        self.lane_id_var = tk.IntVar(value=-1)
        lane_id_spinbox = ttk.Spinbox(param_frame2, from_=-10, to=10, 
                                      textvariable=self.lane_id_var, width=10)
        lane_id_spinbox.pack(side=tk.RIGHT)
        
        # Lane Width
        param_frame3 = ttk.Frame(toolbar)
        param_frame3.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(param_frame3, text="车道宽度(m):").pack(side=tk.LEFT)
        self.lane_width_var = tk.DoubleVar(value=4.0)
        width_spinbox = ttk.Spinbox(param_frame3, from_=2.0, to=10.0, 
                                    textvariable=self.lane_width_var, 
                                    increment=0.5, width=10)
        width_spinbox.pack(side=tk.RIGHT)
        
        ttk.Separator(toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 操作按钮
        ttk.Label(toolbar, text="操作", font=("Arial", 12, "bold")).pack(pady=10)
        
        btn_frame = ttk.Frame(toolbar)
        btn_frame.pack(fill=tk.X, padx=10)
        
        ttk.Button(btn_frame, text="编辑选中路段", 
                  command=self._edit_selected_segment).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="删除选中路段", 
                  command=self._delete_selected_segment).pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="清空所有", 
                  command=self._clear_all).pack(fill=tk.X, pady=5)
        
        ttk.Separator(toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 文件操作
        ttk.Label(toolbar, text="文件", font=("Arial", 12, "bold")).pack(pady=10)
        
        file_frame = ttk.Frame(toolbar)
        file_frame.pack(fill=tk.X, padx=10)
        
        ttk.Button(file_frame, text="加载地图", 
                  command=self._load_map).pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="保存地图", 
                  command=self._save_map).pack(fill=tk.X, pady=5)
        
        # 地图名称
        map_name_frame = ttk.Frame(toolbar)
        map_name_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(map_name_frame, text="地图名称:").pack()
        self.map_name_var = tk.StringVar(value="CustomMap")
        ttk.Entry(map_name_frame, textvariable=self.map_name_var).pack(fill=tk.X)
        
        # 右侧画布容器
        canvas_container = ttk.Frame(main_container)
        canvas_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 顶部信息栏
        info_frame = ttk.Frame(canvas_container)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.info_label = ttk.Label(info_frame, text="准备就绪", relief=tk.SUNKEN)
        self.info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.coord_label = ttk.Label(info_frame, text="坐标: (0, 0)", relief=tk.SUNKEN, width=20)
        self.coord_label.pack(side=tk.RIGHT)
        
        # 画布
        self.canvas = tk.Canvas(canvas_container, bg="white", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        self.canvas.configure(scrollregion=(-5000, -5000, 5000, 5000))
        
        # 初始化画布
        self._draw_grid()
    
    def _bind_events(self):
        """绑定事件"""
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-2>", self._on_middle_mouse_down)
        self.canvas.bind("<B2-Motion>", self._on_middle_mouse_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_mouse_up)
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Delete>", lambda e: self._delete_selected_segment())
    
    def _on_tool_change(self):
        """工具切换时的处理"""
        self.current_tool = self.tool_var.get()
        self.current_segment_points = []
        self._update_info(f"切换到: {self.current_tool}")
        self._redraw_canvas()
    
    def _on_canvas_click(self, event):
        """画布点击事件"""
        # 转换为世界坐标
        world_x, world_y = self._screen_to_world(event.x, event.y)
        
        if self.current_tool == "select":
            # 选择模式
            self._select_segment_at(world_x, world_y)
        
        elif self.current_tool == "line":
            # 直线工具
            self.current_segment_points.append((world_x, world_y))
            self._redraw_canvas()
            self._update_info(f"直线工具: 已添加点 {len(self.current_segment_points)}")
        
        elif self.current_tool == "curve":
            # 曲线工具
            self.current_segment_points.append((world_x, world_y))
            self._redraw_canvas()
            self._update_info(f"曲线工具: 已添加点 {len(self.current_segment_points)}")
    
    def _on_canvas_right_click(self, event):
        """右键点击事件 - 完成当前路段绘制"""
        if self.current_tool in ["line", "curve"] and len(self.current_segment_points) >= 2:
            # 创建路段
            segment = RoadSegment(
                points=self.current_segment_points.copy(),
                road_id=self.road_id_var.get(),
                lane_id=self.lane_id_var.get(),
                width=self.lane_width_var.get()
            )
            self.segments.append(segment)
            
            # 清空当前点
            self.current_segment_points = []
            
            # 自动递增road_id
            self.road_id_var.set(self.road_id_var.get() + 1)
            
            self._redraw_canvas()
            self._update_info(f"完成路段绘制，共 {len(self.segments)} 个路段")
    
    def _on_canvas_motion(self, event):
        """鼠标移动事件"""
        world_x, world_y = self._screen_to_world(event.x, event.y)
        self.mouse_x = world_x
        self.mouse_y = world_y
        
        self.coord_label.config(text=f"坐标: ({world_x:.1f}, {world_y:.1f})")
        
        # 如果正在绘制，显示预览
        if self.current_tool in ["line", "curve"] and self.current_segment_points:
            self._redraw_canvas()
    
    def _on_mouse_wheel(self, event):
        """鼠标滚轮缩放"""
        # 获取鼠标位置
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # 缩放因子
        factor = 1.1 if event.delta > 0 else 0.9
        
        self.scale *= factor
        self.scale = max(0.1, min(10.0, self.scale))  # 限制缩放范围
        
        self._redraw_canvas()
    
    def _on_middle_mouse_down(self, event):
        """中键按下 - 开始平移"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")
    
    def _on_middle_mouse_drag(self, event):
        """中键拖动 - 平移画布"""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.canvas.xview_scroll(int(-dx), "units")
            self.canvas.yview_scroll(int(-dy), "units")
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
    
    def _on_middle_mouse_up(self, event):
        """中键释放 - 结束平移"""
        self.is_panning = False
        self.canvas.config(cursor="crosshair")
    
    def _on_escape(self, event):
        """ESC键 - 取消当前操作"""
        self.current_segment_points = []
        self.selected_segment = None
        self._redraw_canvas()
        self._update_info("操作已取消")
    
    def _screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """屏幕坐标转世界坐标"""
        canvas_x = self.canvas.canvasx(screen_x)
        canvas_y = self.canvas.canvasy(screen_y)
        
        # 画布中心对应世界坐标原点
        world_x = canvas_x / self.scale
        world_y = canvas_y / self.scale
        
        return world_x, world_y
    
    def _world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """世界坐标转屏幕坐标"""
        canvas_x = world_x * self.scale
        canvas_y = world_y * self.scale
        
        return canvas_x, canvas_y
    
    def _draw_grid(self):
        """绘制网格"""
        # 清除旧网格
        self.canvas.delete("grid")
        
        # 获取可见区域
        x1 = self.canvas.canvasx(0)
        y1 = self.canvas.canvasy(0)
        x2 = self.canvas.canvasx(self.canvas.winfo_width())
        y2 = self.canvas.canvasy(self.canvas.winfo_height())
        
        # 绘制网格线
        grid_step = self.grid_size * self.scale
        
        # 垂直线
        x = int(x1 / grid_step) * grid_step
        while x < x2:
            self.canvas.create_line(x, y1, x, y2, fill="lightgray", tags="grid")
            x += grid_step
        
        # 水平线
        y = int(y1 / grid_step) * grid_step
        while y < y2:
            self.canvas.create_line(x1, y, x2, y, fill="lightgray", tags="grid")
            y += grid_step
        
        # 绘制坐标轴
        origin_x, origin_y = self._world_to_screen(0, 0)
        self.canvas.create_line(x1, origin_y, x2, origin_y, fill="red", width=2, tags="grid")
        self.canvas.create_line(origin_x, y1, origin_x, y2, fill="red", width=2, tags="grid")
    
    def _redraw_canvas(self):
        """重绘画布"""
        # 清除所有内容
        self.canvas.delete("all")
        
        # 绘制网格
        self._draw_grid()
        
        # 绘制所有路段
        for segment in self.segments:
            self._draw_segment(segment)
        
        # 绘制当前正在绘制的路段
        if self.current_segment_points:
            self._draw_current_segment()
    
    def _draw_segment(self, segment: RoadSegment):
        """绘制路段"""
        if len(segment.points) < 2:
            return
        
        # 转换为屏幕坐标
        screen_points = []
        for px, py in segment.points:
            sx, sy = self._world_to_screen(px, py)
            screen_points.extend([sx, sy])
        
        # 选择颜色
        color = "blue" if segment.selected else "black"
        width = 3 if segment.selected else 2
        
        # 绘制路径
        segment.canvas_id = self.canvas.create_line(
            *screen_points, 
            fill=color, 
            width=width,
            tags="segment"
        )
        
        # 绘制路段信息
        if segment.selected:
            mid_idx = len(segment.points) // 2
            mid_x, mid_y = segment.points[mid_idx]
            sx, sy = self._world_to_screen(mid_x, mid_y)
            
            text = f"R{segment.road_id} L{segment.lane_id}"
            self.canvas.create_text(
                sx, sy - 15,
                text=text,
                fill="blue",
                font=("Arial", 10, "bold"),
                tags="segment"
            )
        
        # 绘制宽度指示
        if segment.selected and len(segment.points) >= 2:
            # 在第一段绘制宽度指示
            p1 = segment.points[0]
            p2 = segment.points[1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)
            
            if length > 0.001:
                perp_x = -dy / length * segment.width / 2
                perp_y = dx / length * segment.width / 2
                
                # 绘制宽度边界
                for points in [
                    [(p1[0] - perp_x, p1[1] - perp_y), (p2[0] - perp_x, p2[1] - perp_y)],
                    [(p1[0] + perp_x, p1[1] + perp_y), (p2[0] + perp_x, p2[1] + perp_y)]
                ]:
                    screen_pts = []
                    for px, py in points:
                        sx, sy = self._world_to_screen(px, py)
                        screen_pts.extend([sx, sy])
                    
                    self.canvas.create_line(
                        *screen_pts,
                        fill="cyan",
                        width=1,
                        dash=(5, 5),
                        tags="segment"
                    )
    
    def _draw_current_segment(self):
        """绘制当前正在绘制的路段"""
        points = self.current_segment_points.copy()
        
        # 添加鼠标当前位置作为预览
        if self.current_tool in ["line", "curve"]:
            points.append((self.mouse_x, self.mouse_y))
        
        if len(points) < 2:
            # 只绘制点
            for px, py in points:
                sx, sy = self._world_to_screen(px, py)
                self.canvas.create_oval(
                    sx - 5, sy - 5, sx + 5, sy + 5,
                    fill="red",
                    tags="current"
                )
            return
        
        # 转换为屏幕坐标
        screen_points = []
        for px, py in points:
            sx, sy = self._world_to_screen(px, py)
            screen_points.extend([sx, sy])
        
        # 绘制路径
        self.canvas.create_line(
            *screen_points,
            fill="red",
            width=2,
            dash=(5, 5),
            tags="current"
        )
        
        # 绘制点
        for px, py in points:
            sx, sy = self._world_to_screen(px, py)
            self.canvas.create_oval(
                sx - 5, sy - 5, sx + 5, sy + 5,
                fill="red",
                tags="current"
            )
    
    def _select_segment_at(self, x: float, y: float):
        """在指定位置选择路段"""
        # 取消之前的选择
        if self.selected_segment:
            self.selected_segment.selected = False
        
        self.selected_segment = None
        
        # 查找点击位置的路段
        for segment in self.segments:
            if segment.contains_point(x, y):
                segment.selected = True
                self.selected_segment = segment
                self._update_info(f"已选择路段: Road ID={segment.road_id}, Lane ID={segment.lane_id}")
                break
        
        if not self.selected_segment:
            self._update_info("未选择任何路段")
        
        self._redraw_canvas()
    
    def _edit_selected_segment(self):
        """编辑选中的路段"""
        if not self.selected_segment:
            messagebox.showwarning("警告", "请先选择一个路段")
            return
        
        # 创建编辑对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("编辑路段")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Road ID
        ttk.Label(dialog, text="Road ID:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        road_id_var = tk.IntVar(value=self.selected_segment.road_id)
        ttk.Spinbox(dialog, from_=0, to=10000, textvariable=road_id_var).grid(row=0, column=1, padx=10, pady=10)
        
        # Lane ID
        ttk.Label(dialog, text="Lane ID:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        lane_id_var = tk.IntVar(value=self.selected_segment.lane_id)
        ttk.Spinbox(dialog, from_=-10, to=10, textvariable=lane_id_var).grid(row=1, column=1, padx=10, pady=10)
        
        # Width
        ttk.Label(dialog, text="车道宽度 (m):").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        width_var = tk.DoubleVar(value=self.selected_segment.width)
        ttk.Spinbox(dialog, from_=2.0, to=10.0, textvariable=width_var, increment=0.5).grid(row=2, column=1, padx=10, pady=10)
        
        def apply_changes():
            self.selected_segment.road_id = road_id_var.get()
            self.selected_segment.lane_id = lane_id_var.get()
            self.selected_segment.width = width_var.get()
            self._redraw_canvas()
            self._update_info(f"已更新路段: Road ID={self.selected_segment.road_id}, Lane ID={self.selected_segment.lane_id}")
            dialog.destroy()
        
        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="确定", command=apply_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _delete_selected_segment(self):
        """删除选中的路段"""
        if not self.selected_segment:
            messagebox.showwarning("警告", "请先选择一个路段")
            return
        
        if messagebox.askyesno("确认", "确定要删除选中的路段吗？"):
            self.segments.remove(self.selected_segment)
            self.selected_segment = None
            self._redraw_canvas()
            self._update_info(f"已删除路段，剩余 {len(self.segments)} 个路段")
    
    def _clear_all(self):
        """清空所有路段"""
        if not self.segments:
            return
        
        if messagebox.askyesno("确认", f"确定要清空所有 {len(self.segments)} 个路段吗？"):
            self.segments = []
            self.selected_segment = None
            self.current_segment_points = []
            self._redraw_canvas()
            self._update_info("已清空所有路段")
    
    def _save_map(self):
        """保存地图到JSON文件"""
        if not self.segments:
            messagebox.showwarning("警告", "没有可保存的路段")
            return
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{self.map_name_var.get()}.json"
        )
        
        if not file_path:
            return
        
        # 生成quads数据
        quads = []
        poly_id = 0
        
        for segment in self.segments:
            segment_quads = segment.to_quads(poly_id)
            quads.extend(segment_quads)
            poly_id += len(segment_quads)
        
        # 构建地图数据
        map_data = {
            "map_name": self.map_name_var.get(),
            "quads": quads,
            "traffic_controls": [],
            "global_w_lane_waypoints": []
        }
        
        # 保存到文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("成功", f"地图已保存到: {file_path}\n共 {len(quads)} 个四边形")
            self._update_info(f"地图已保存: {len(quads)} 个四边形")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def _load_map(self):
        """从JSON文件加载地图"""
        # 如果有未保存的数据，提示用户
        if self.segments and not messagebox.askyesno("确认", "当前有未保存的数据，确定要加载新地图吗？"):
            return
        
        # 选择文件
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
            
            # 清空当前数据
            self.segments = []
            self.selected_segment = None
            self.current_segment_points = []
            
            # 设置地图名称
            self.map_name_var.set(map_data.get("map_name", "LoadedMap"))
            
            # 解析quads
            quads = map_data.get("quads", [])
            
            if not quads:
                messagebox.showwarning("警告", "地图文件中没有找到quads数据")
                return
            
            # 将quads重建为路段
            # 这是一个简化的实现，将每个quad的中心线作为路段的一部分
            road_segments_dict = {}  # {(road_id, lane_id): [points]}
            
            for quad in quads:
                road_id = quad.get("road_id", 0)
                lane_id = quad.get("lane_id", -1)
                vertices = quad.get("vertices", [])
                
                if len(vertices) < 4:
                    continue
                
                # 计算quad的中心线点（假设vertices顺序为逆时针）
                p1 = ((vertices[0]["x"] + vertices[3]["x"]) / 2,
                      (vertices[0]["y"] + vertices[3]["y"]) / 2)
                p2 = ((vertices[1]["x"] + vertices[2]["x"]) / 2,
                      (vertices[1]["y"] + vertices[2]["y"]) / 2)
                
                # 计算宽度
                width = math.sqrt((vertices[1]["x"] - vertices[0]["x"])**2 + 
                                 (vertices[1]["y"] - vertices[0]["y"])**2)
                
                key = (road_id, lane_id)
                if key not in road_segments_dict:
                    road_segments_dict[key] = {"points": [p1], "width": width}
                
                road_segments_dict[key]["points"].append(p2)
            
            # 创建路段对象
            for (road_id, lane_id), data in road_segments_dict.items():
                segment = RoadSegment(
                    points=data["points"],
                    road_id=road_id,
                    lane_id=lane_id,
                    width=data["width"]
                )
                self.segments.append(segment)
            
            self._redraw_canvas()
            
            messagebox.showinfo("成功", f"地图已加载\n共 {len(quads)} 个四边形\n重建为 {len(self.segments)} 个路段")
            self._update_info(f"已加载地图: {len(self.segments)} 个路段")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_info(self, text: str):
        """更新信息栏"""
        self.info_label.config(text=text)


def main():
    """主函数"""
    root = tk.Tk()
    app = RoadNetworkEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()

