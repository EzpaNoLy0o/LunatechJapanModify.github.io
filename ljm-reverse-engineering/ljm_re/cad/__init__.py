"""CADバックエンド（OpenCASCADE / OCP）.

CadQuery に依存せず OCP を直接使う独立バックエンドとして実装している。
OCP が利用できない環境では available() が False を返し、CAD機能のみ無効になる。
"""
