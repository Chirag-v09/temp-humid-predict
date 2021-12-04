# -*- coding: utf-8 -*-

from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements


class payload(BaseModel):
    obs: int

