#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的日志模块
"""
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('GDesigner')

