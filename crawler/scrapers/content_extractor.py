"""
고품질 콘텐츠 추출기
Git 명령어와 Python 함수의 설명, 사용법, 예제를 정확하게 추출
"""
from bs4 import BeautifulSoup, Comment
from typing import Dict, List, Optional
import re
from config.settings_extended import CONTENT_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class QualityContentExtractor:
    """
    고품질 콘텐츠 추출
    - 노이즈 완전 제거
    - 구조화된 섹션 추출
    - 코드 예제 분리
    """
    
    def __init__(self):
        self.exclude_selectors = CONTENT_CONFIG["exclude_selectors"]
        self.include_sections = CONTENT_CONFIG["include_sections"]
        self.min_text_length = CONTENT_CONFIG["min_text_length"]
        self.code_selectors = CONTENT_CONFIG["code_selectors"]
    
    def extract(self, html: str, url: str, doc_type: str = "unknown") -> Optional[Dict]:
        """
        HTML에서 고품질 콘텐츠 추출
        
        Args:
            html: HTML 문자열
            url: 페이지 URL
            doc_type: "git" 또는 "python"
        
        Returns:
            {
                'title': '명령어 또는 함수명',
                'summary': '간단 설명',
                'sections': [
                    {
                        'header': '섹션 제목',
                        'content': '설명 텍스트',
                        'code_examples': ['코드1', 'code2'],
                        'usage_examples': ['사용 예제'],
                        'metadata': {...}
                    }
                ],
                'url': str,
                'doc_type': str
            }
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # 1. 모든 노이즈 제거
            self._remove_all_noise(soup)
            
            # 2. 메인 콘텐츠 찾기
            main_content = self._find_main_content(soup)
            if not main_content:
                logger.warning(f"메인 콘텐츠 없음: {url}")
                return None
            
            # 3. 제목 및 요약
            title = self._extract_title(soup)
            summary = self._extract_summary(main_content)
            
            # 4. 섹션 추출
            sections = self._extract_quality_sections(main_content, doc_type)
            
            # 5. 품질 검증
            if not sections:
                logger.warning(f"섹션 없음: {url}")
                return None
            
            # 빈 섹션 제거
            sections = [s for s in sections if s['content'].strip()]
            
            if not sections:
                return None
            
            return {
                'title': title,
                'summary': summary,
                'sections': sections,
                'url': url,
                'doc_type': doc_type,
                'total_sections': len(sections)
            }
            
        except Exception as e:
            logger.error(f"콘텐츠 추출 오류: {url} - {e}")
            return None
    
    def _remove_all_noise(self, soup: BeautifulSoup):
        """모든 노이즈 요소 제거"""
        # 주석
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # 스크립트, 스타일
        for tag in soup(['script', 'style', 'iframe', 'noscript', 'svg']):
            tag.decompose()
        
        # 지정 선택자
        for selector in self.exclude_selectors:
            try:
                for element in soup.select(selector):
                    element.decompose()
            except Exception:
                pass
        
        # 빈 태그
        for tag in soup.find_all():
            if not tag.get_text(strip=True) and not tag.find(['img', 'pre', 'code']):
                tag.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup):
        """메인 콘텐츠 영역 찾기"""
        # Git 문서 전용: #main 우선
        main = soup.find(id='main')
        if main and len(main.get_text(strip=True)) > 200:
            return main
        
        # include_sections 순서대로
        for selector in self.include_sections:
            try:
                content = soup.select_one(selector)
                if content and len(content.get_text(strip=True)) > 200:
                    return content
            except:
                pass
        
        # role="main"
        main = soup.find(attrs={"role": "main"})
        if main:
            return main
        
        # article
        article = soup.find('article')
        if article:
            return article
        
        # body
        return soup.find('body')

    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """제목 추출"""
        # h1
        h1 = soup.find('h1')
        if h1:
            return self._clean_text(h1.get_text())
        
        # title 태그
        title = soup.find('title')
        if title:
            text = title.get_text()
            # 사이트 이름 제거
            text = re.split(r' - | \| ', text)[0]
            return self._clean_text(text)
        
        return "Untitled"
    
    def _extract_summary(self, content) -> str:
        """페이지 요약"""
        summary_parts = []
        
        for elem in content.find_all(['p', 'div'], limit=5):
            text = self._clean_text(elem.get_text())
            if len(text) > 50:
                summary_parts.append(text)
                if len(summary_parts) >= 2:
                    break
        
        return '\n\n'.join(summary_parts)[:500]
    
    def _extract_quality_sections(self, content, doc_type: str) -> List[Dict]:
        """명령어/함수별 섹션 추출"""
        sections = []
        current_section = self._new_section()
        
        # h2, h3 우선 처리
        for element in content.find_all(['h2', 'h3', 'h4', 'p', 'pre', 'code', 'ul', 'ol', 'dl', 'table', 'div']):
            tag_name = element.name
            
            # 헤더 = 새 섹션
            if tag_name in ['h2', 'h3']:  # h2, h3만 섹션으로
                if self._is_quality_section(current_section):
                    sections.append(self._finalize_section(current_section))
                
                header_text = self._clean_text(element.get_text())
                current_section = self._new_section()
                current_section['header'] = header_text
                current_section['level'] = tag_name

            
            # 코드 블록
            elif tag_name == 'pre':
                code_text = element.get_text(strip=True)
                if len(code_text) > 5:
                    lang = self._detect_code_language(element, doc_type)
                    current_section['code_examples'].append({
                        'code': code_text,
                        'language': lang
                    })
            
            # 인라인 코드
            elif tag_name == 'code' and element.parent.name != 'pre':
                code_text = element.get_text(strip=True)
                if len(code_text) > 2 and '`' not in code_text:
                    current_section['inline_codes'].append(code_text)
            
            # 리스트
            elif tag_name in ['ul', 'ol']:
                list_items = []
                for li in element.find_all('li', recursive=False):
                    item_text = self._clean_text(li.get_text())
                    if len(item_text) > 10:
                        list_items.append(item_text)
                
                if list_items:
                    current_section['lists'].append({
                        'type': tag_name,
                        'items': list_items
                    })
            
            # Definition list
            elif tag_name == 'dl':
                definitions = self._extract_definitions(element)
                if definitions:
                    current_section['definitions'].extend(definitions)
            
            # 테이블
            elif tag_name == 'table':
                table_data = self._extract_table(element)
                if table_data:
                    current_section['tables'].append(table_data)
            
            # 텍스트
            elif tag_name == 'p':
                text = self._clean_text(element.get_text())
                if len(text) >= self.min_text_length:
                    current_section['paragraphs'].append(text)
            
            # 특수 div
            elif tag_name == 'div':
                special_content = self._extract_special_div(element)
                if special_content:
                    current_section['special_blocks'].append(special_content)
        
        # 마지막 섹션
        if self._is_quality_section(current_section):
            sections.append(self._finalize_section(current_section))
        
        return sections
    
    def _new_section(self) -> Dict:
        """섹션 초기화"""
        return {
            'header': '',
            'paragraphs': [],
            'code_examples': [],
            'inline_codes': [],
            'lists': [],
            'definitions': [],
            'tables': [],
            'special_blocks': [],
            'level': 'h2'
        }
    
    def _is_quality_section(self, section: Dict) -> bool:
        """
        섹션 품질 체크 (더 관대하게 수정)
        - 헤더 없어도 콘텐츠가 충분하면 OK
        - 최소 50자 이상의 텍스트 또는 코드 예제
        """
        has_header = bool(section['header'])
        
        # 콘텐츠 길이 계산
        total_text_length = sum(len(p) for p in section['paragraphs'])
        total_code_length = sum(len(c.get('code', '')) for c in section['code_examples'])
        
        has_content = (
            total_text_length >= 50 or  # 50자 이상
            total_code_length >= 20 or  # 코드 20자 이상
            len(section['paragraphs']) > 0 or
            len(section['code_examples']) > 0 or
            len(section['lists']) > 0 or
            len(section['definitions']) > 0 or
            len(section['tables']) > 0
        )
        return has_header or has_content
    
    def _finalize_section(self, section: Dict) -> Dict:
        """섹션 최종 정리"""
        content_parts = []
        
        # 텍스트
        if section['paragraphs']:
            content_parts.extend(section['paragraphs'])
        
        # 리스트
        for list_info in section['lists']:
            list_text = '\n'.join([f"• {item}" for item in list_info['items']])
            content_parts.append(list_text)
        
        # Definition
        if section['definitions']:
            def_text = '\n'.join([
                f"**{d['term']}**: {d['definition']}" 
                for d in section['definitions']
            ])
            content_parts.append(def_text)
        
        # 테이블
        for table in section['tables']:
            table_text = self._format_table(table)
            content_parts.append(table_text)
        
        # 특수 블록
        for block in section['special_blocks']:
            content_parts.append(f"[{block['type']}] {block['content']}")
        
        # 최종 콘텐츠
        main_content = '\n\n'.join(content_parts)
        
        # 코드 예제 정리
        code_examples = []
        usage_examples = []

        for code_info in section['code_examples']:
            # 실제 코드를 포맷팅하여 저장
            lang = code_info['language']
            code = code_info['code']
            code_str = f"```{lang}\n{code}\n```"

            if self._is_usage_example(code):
                usage_examples.append(code_str)
            else:
                code_examples.append(code_str)
        
        # 인라인 코드
        if section['inline_codes']:
            inline_text = "관련 명령어/함수: " + ", ".join(
                [f"`{code}`" for code in section['inline_codes'][:10]]
            )
            content_parts.append(inline_text)
        
        return {
            'header': section['header'],
            'content': main_content,
            'code_examples': code_examples,
            'usage_examples': usage_examples,
            'metadata': {
                'level': section['level'],
                'has_code': len(code_examples) > 0,
                'has_usage': len(usage_examples) > 0,
                'has_table': len(section['tables']) > 0,
                'paragraph_count': len(section['paragraphs']),
                'content_length': len(main_content)
            }
        }
    
    def _detect_code_language(self, element, doc_type: str) -> str:
        """코드 언어 감지"""
        classes = element.get('class', [])
        for cls in classes:
            if 'language-' in cls:
                return cls.replace('language-', '')
            if 'lang-' in cls:
                return cls.replace('lang-', '')
        
        if doc_type == 'git':
            return 'bash'
        elif doc_type == 'python':
            return 'python'
        
        return ''
    
    def _is_usage_example(self, code: str) -> bool:
        """사용 예제 판별"""
        usage_indicators = ['$', '>>>', '# Example', '# Usage', '# How to', 'python', 'git']
        return any(indicator in code for indicator in usage_indicators)
    
    def _extract_definitions(self, dl_element) -> List[Dict]:
        """Definition list 추출"""
        definitions = []
        dts = dl_element.find_all('dt')
        dds = dl_element.find_all('dd')
        
        for dt, dd in zip(dts, dds):
            term = self._clean_text(dt.get_text())
            definition = self._clean_text(dd.get_text())
            
            if term and definition:
                definitions.append({
                    'term': term,
                    'definition': definition
                })
        
        return definitions
    
    def _extract_table(self, table_element) -> Optional[Dict]:
        """테이블 추출"""
        try:
            rows = []
            headers = []
            
            thead = table_element.find('thead')
            if thead:
                header_cells = thead.find_all(['th', 'td'])
                headers = [self._clean_text(cell.get_text()) for cell in header_cells]
            
            tbody = table_element.find('tbody') or table_element
            for tr in tbody.find_all('tr'):
                cells = tr.find_all(['td', 'th'])
                if cells:
                    row = [self._clean_text(cell.get_text()) for cell in cells]
                    rows.append(row)
            
            if rows:
                return {'headers': headers, 'rows': rows}
        except:
            pass
        
        return None
    
    def _format_table(self, table: Dict) -> str:
        """테이블 포맷팅"""
        lines = []
        
        if table['headers']:
            lines.append(' | '.join(table['headers']))
            lines.append(' | '.join(['---'] * len(table['headers'])))
        
        for row in table['rows']:
            lines.append(' | '.join(row))
        
        return '\n'.join(lines)
    
    def _extract_special_div(self, div_element) -> Optional[Dict]:
        """특수 div 추출"""
        classes = ' '.join(div_element.get('class', []))
        special_types = ['example', 'note', 'warning', 'tip', 'admonition', 'alert', 'info']
        
        for special_type in special_types:
            if special_type in classes.lower():
                text = self._clean_text(div_element.get_text())
                if len(text) > 30:
                    return {
                        'type': special_type.capitalize(),
                        'content': text
                    }
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
        return text.strip()
