(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var root = document.documentElement;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  root.classList.remove('no-js');
  var store = { get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };

  /* ---------- language ---------- */
  var lang = 'zh';
  $$('[data-en]').forEach(function (el) { el.setAttribute('data-zh', el.innerHTML); });
  var hooks = [];
  function setLang(l) {
    lang = l;
    root.lang = l === 'zh' ? 'zh-CN' : 'en';
    document.body.classList.toggle('zh', l === 'zh');
    $$('[data-en]').forEach(function (el) {
      var v = el.getAttribute(l === 'zh' ? 'data-zh' : 'data-en');
      if (el.tagName === 'IMG') el.alt = v; else el.innerHTML = v;
    });
    var b = $('#langbtn'); if (b) { b.textContent = l === 'zh' ? 'EN' : '中文'; b.setAttribute('aria-label', l === 'zh' ? 'Switch to English' : '切换到中文'); }
    document.title = l === 'zh' ? 'Allocation Before the Clock：职业 LoL 资源团前的站位研究' : 'Allocation Before the Clock: how pro League of Legends teams set up before objectives';
    store.set('lang', l);
    hooks.forEach(function (h) { h(); });
  }
  var L = function (o) { return o[lang]; };
  $('#langbtn').addEventListener('click', function () { setLang(lang === 'zh' ? 'en' : 'zh'); });

  /* ---------- reveal ---------- */
  if ('IntersectionObserver' in window && !reduce) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    $$('.rv, .chart').forEach(function (el) { io.observe(el); });
  } else { $$('.rv, .chart').forEach(function (el) { el.classList.add('in'); }); }

  /* ---------- objective selector on the map ---------- */
  var map = JSON.parse($('#mapdata').textContent);
  var OBJ = [
    { k: 'DRAGON', zh: '元素龙', en: 'Dragon', t: '5:00', n: '773', d: 49,
      zhS: '河道里的老朋友,一局会刷新好几次。我们记录了 773 次。', enS: 'The river regular; it respawns several times a game. We logged 773 spawns.' },
    { k: 'HORDE', zh: '虚空巢虫', en: 'Void Grubs', t: '8:00', n: '164', d: 21,
      zhS: '三只一组,和峡谷先锋共用同一个坑。刷出来后很快就被清掉。', enS: 'Three at a time, in the same pit as the Rift Herald. Cleared very quickly.' },
    { k: 'RIFTHERALD', zh: '峡谷先锋', en: 'Rift Herald', t: '15:00', n: '164', d: 50,
      zhS: '和巢虫同一个坑(上半区)。', enS: 'Shares the upper pit with the grubs.' },
    { k: 'ATAKHAN', zh: '亚塔坎', en: 'Atakhan', t: '20:00', n: '164', d: 109,
      zhS: '2025 赛季新加入的资源,出现在河道一侧;拿它的队伍常常不是先到的那队。', enS: 'New in the 2025 season, in one river lane. The team that arrives first often does not get it.' },
    { k: 'BARON_NASHOR', zh: '纳什男爵', en: 'Baron Nashor', t: '25:00', n: '224', d: 117,
      zhS: '最大的一个坑,赢下它通常意味着可以推进。', enS: 'The biggest pit; winning it usually opens the way to push.' }
  ];
  var bl = $('#blds');
  if (bl) map.b.forEach(function (b) {
    var c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', b[2]); c.setAttribute('cy', 150 - (150 - b[3])); c.setAttribute('r', b[1] === 'I' ? 2.1 : 1.5);
    c.setAttribute('fill', b[0] === 100 ? 'var(--blue)' : 'var(--red)'); bl.appendChild(c);
  });
  var sel = $('#objsel'), fact = $('#objfact');
  var ringG = $('#pitg');
  function paintObj(i) {
    var o = OBJ[i], p = map.pits[o.k];
    ringG.setAttribute('transform', 'translate(' + p[0].toFixed(1) + ' ' + p[1].toFixed(1) + ')');
    $$('button', sel).forEach(function (b, j) { b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
    sel.dataset.i = i;
    paintFact();
  }
  function paintFact() {
    var o = OBJ[+sel.dataset.i || 0];
    fact.innerHTML = lang === 'zh'
      ? '<b>' + o.t + '</b> 第一次出现。中位数在刷新后约 <b>' + o.d + ' 秒</b>被击杀。<br>' + o.zhS
      : 'First appears at <b>' + o.t + '</b>. Median time from spawn to kill: <b>' + o.d + ' s</b>.<br>' + o.enS;
  }
  OBJ.forEach(function (o, i) {
    var li = document.createElement('li'), b = document.createElement('button');
    b.type = 'button'; b.setAttribute('data-en', o.en); b.setAttribute('data-zh', o.zh); b.textContent = o.zh;
    b.addEventListener('click', function () { paintObj(i); });
    li.appendChild(b); sel.appendChild(li);
  });
  hooks.push(function () { $$('button', sel).forEach(function (b, j) { b.textContent = L({ zh: OBJ[j].zh, en: OBJ[j].en }); }); paintFact(); });
  paintObj(0);

  /* ---------- accordion ---------- */
  var pans = $$('.pan');
  function openPan(p) { pans.forEach(function (x) { x.setAttribute('aria-expanded', x === p ? 'true' : 'false'); }); }
  pans.forEach(function (p) { p.addEventListener('click', function () { openPan(p); }); });
  openPan(pans[0]);

  /* ---------- pattern viewer ---------- */
  var PD = JSON.parse($('#patdata').textContent);
  var HOME = { T: [13, 78], J: [40, 100], M: [75, 75], B: [72, 138], S: [82, 134] };
  var PIT = map.pits.BARON_NASHOR;
  var pull = { 1: 0, 2: 0.3, 3: 0.6, 4: 0.85, 5: 0.95 };
  var NAME = { T: ['上单', 'Top'], J: ['打野', 'Jungle'], M: ['中单', 'Mid'], B: ['下路', 'Bot'], S: ['辅助', 'Support'] };
  var plG = $('#players'), pats = $('#pats'), slider = $('#tslider'), tlabel = $('#tlabel'), pread = $('#pread'), pdesc = $('#pdesc'), pw = $('#pweights');
  var roles = {};
  ['T', 'J', 'M', 'B', 'S'].forEach(function (r) {
    var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'pl'); g.innerHTML = '<circle r="4.3"></circle><text>' + r + '</text>';
    plG.appendChild(g); roles[r] = g;
  });
  var curP = 0;
  function parse(s) { return s.split('|').map(function (b) { return b.replace(/\s/g, '').split(''); }); }
  function offsetLabel(i) { var o = -120 + 15 * i; return o === 0 ? L({ zh: '刷新的一刻', en: 'the moment of spawn' }) : (o < 0 ? L({ zh: '刷新前 ' + (-o) + ' 秒', en: (-o) + ' s before spawn' }) : L({ zh: '刷新后 ' + o + ' 秒', en: o + ' s after spawn' })); }
  function paintPat() {
    var P = PD[curP], i = +slider.value, blocks = parse(P.path[i]);
    blocks.forEach(function (b) {
      var cx = 0, cy = 0; b.forEach(function (r) { cx += HOME[r][0]; cy += HOME[r][1]; });
      cx /= b.length; cy /= b.length;
      var f = pull[b.length]; cx += (PIT[0] - cx) * f; cy += (PIT[1] - cy) * f;
      b.forEach(function (r, k) {
        var a = (k / b.length) * Math.PI * 2 + 0.6, rad = b.length > 1 ? 3.1 + b.length * 0.35 : 0;
        roles[r].setAttribute('transform', 'translate(' + (cx + Math.cos(a) * rad).toFixed(1) + ' ' + (cy + Math.sin(a) * rad).toFixed(1) + ')');
      });
    });
    tlabel.textContent = offsetLabel(i);
    var parts = blocks.map(function (b) { return b.map(function (r) { return L({ zh: NAME[r][0], en: NAME[r][1] }); }).join(' + '); });
    pread.innerHTML = lang === 'zh'
      ? '这一刻分成 <b>' + blocks.length + ' 组</b>:' + parts.join(' · ') + '。'
      : 'At this moment the team is in <b>' + blocks.length + ' group' + (blocks.length > 1 ? 's' : '') + '</b>: ' + parts.join(' · ') + '.';
  }
  function paintPatText() {
    var P = PD[curP];
    pdesc.innerHTML = L(P.desc);
    pw.innerHTML = '<span>' + L({ zh: '早期河道资源 ', en: 'Early river ' }) + '<b>' + P.w[0] + '%</b></span><span>' + L({ zh: '亚塔坎 ', en: 'Atakhan ' }) + '<b>' + P.w[1] + '%</b></span><span>' + L({ zh: '后期资源 ', en: 'Late objectives ' }) + '<b>' + P.w[2] + '%</b></span>';
  }
  PD.forEach(function (P, i) {
    var b = document.createElement('button'); b.type = 'button';
    b.setAttribute('data-en', P.name.en); b.setAttribute('data-zh', P.name.zh); b.textContent = P.name.zh;
    b.addEventListener('click', function () { curP = i; $$('button', pats).forEach(function (x, j) { x.setAttribute('aria-pressed', j === i ? 'true' : 'false'); }); paintPatText(); paintPat(); });
    pats.appendChild(b);
  });
  $$('button', pats)[0].setAttribute('aria-pressed', 'true');
  slider.addEventListener('input', paintPat);
  slider.value = 8;
  hooks.push(function () { $$('button', pats).forEach(function (b, j) { b.textContent = L(PD[j].name); }); paintPatText(); paintPat(); });
  paintPatText(); paintPat();

  /* ---------- figure gallery ---------- */
  var tabs = $$('.gal .tabs button'), imgs = $$('.frame img'), caps = $$('.cap > div');
  var gi = 0, touched = false, timer;
  function showFig(i) {
    gi = i;
    tabs.forEach(function (t, j) { t.setAttribute('aria-selected', j === i ? 'true' : 'false'); });
    imgs.forEach(function (m, j) { m.classList.toggle('on', j === i); });
    caps.forEach(function (c, j) { c.hidden = j !== i; });
  }
  tabs.forEach(function (t, i) { t.addEventListener('click', function () { touched = true; showFig(i); }); });
  showFig(0);
  if (!reduce) {
    var gal = $('.gal'), visible = false;
    if ('IntersectionObserver' in window) new IntersectionObserver(function (e) { visible = e[0].isIntersecting; }, { threshold: 0.3 }).observe(gal);
    timer = setInterval(function () { if (!touched && visible && !document.hidden) showFig((gi + 1) % tabs.length); }, 6500);
  }

  /* ---------- FAQ ---------- */
  var qs = $$('.card > button');
  qs.forEach(function (b) { b.addEventListener('click', function () {
    var open = b.getAttribute('aria-expanded') === 'true';
    qs.forEach(function (x) { x.setAttribute('aria-expanded', 'false'); });
    b.setAttribute('aria-expanded', open ? 'false' : 'true');
  }); });
  qs[0].setAttribute('aria-expanded', 'true');

  var saved = store.get('lang');
  setLang(saved === 'en' ? 'en' : 'zh');
})();
