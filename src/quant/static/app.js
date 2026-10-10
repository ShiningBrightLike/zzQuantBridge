(function () {
  function runParams() {
    var value = function (id) { var node = document.querySelector(id); return node ? node.value.trim() : ''; };
    var params = {};
    var symbols = value('#run-symbols');
    if (symbols) params.symbols = symbols.split(/[,，\s]+/).filter(Boolean);
    ['start', 'end', 'file'].forEach(function (key) {
      var item = value('#run-' + key);
      if (item) params[key] = item;
    });
    var provider = value('#run-provider');
    if (provider) params.provider = provider;
    return params;
  }
  function requestRun(kind) {
    return fetch('/api/runs/' + kind, { method: 'POST', headers: { 'X-CSRF-Token': window.quantCsrf, 'Content-Type': 'application/json' }, body: JSON.stringify(runParams()) })
      .then(function (response) { return response.json().then(function (body) { return { ok: response.ok, body: body }; }); });
  }
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-run-kind]').forEach(function (button) {
      button.addEventListener('click', function () {
        var message = document.querySelector('#job-message');
        button.disabled = true;
        requestRun(button.dataset.runKind).then(function (result) {
          if (!result.ok) throw new Error(result.body.detail || '任务启动失败');
          message.innerHTML = '<a href="/runs/' + result.body.run_id + '">' + result.body.run_id + '</a> 已排队';
        }).catch(function (error) { message.textContent = error.message; }).finally(function () { button.disabled = false; });
      });
    });
    var clearButton = document.querySelector('#clear-history');
    if (clearButton) {
      clearButton.addEventListener('click', function () {
        var message = document.querySelector('#clear-message') || { textContent: '' };
        var output = document.querySelector('#clear-result');
        var checked = function (id) { var node = document.querySelector(id); return node ? node.checked : false; };
        var value = function (id) { var node = document.querySelector(id); return node ? node.value.trim() : ''; };
        try {
          var scopes = [];
          if (checked('#scope-runs')) scopes.push('runs');
          if (checked('#scope-data')) scopes.push('data');
          if (checked('#scope-database')) scopes.push('database');
          var keepDays = value('#keep-days');
          var token = value('#confirm-token');
          if (!scopes.length) { message.textContent = '请至少勾选一项'; return; }
          if (token !== 'CLEAR') { message.textContent = '请输入 CLEAR 以确认'; return; }
          var payload = { scopes: scopes, confirm: token };
          if (keepDays) payload.keep_days = Number(keepDays);
          clearButton.disabled = true;
          message.textContent = '正在删除…';
        } catch (error) {
          message.textContent = '脚本错误：' + error.message;
          return;
        }
        fetch('/api/history/clear', { method: 'POST', headers: { 'X-CSRF-Token': window.quantCsrf, 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
          .then(function (response) { return response.json().then(function (body) { return { ok: response.ok, body: body }; }); })
          .then(function (response) {
            if (!response.ok) throw new Error(typeof response.body.detail === 'string' ? response.body.detail : '清除失败');
            output.textContent = JSON.stringify(response.body, null, 2);
            var freedMb = ((response.body.bytes_freed || 0) / (1024 * 1024)).toFixed(2);
            message.textContent = '已删除 ' + response.body.total + ' 项，释放 ' + freedMb + ' MB';
          })
          .catch(function (error) { message.textContent = '清除失败：' + error.message; })
          .finally(function () { clearButton.disabled = false; });
      });
    }
    var form = document.querySelector('#reconcile-form');
    if (!form) return;
    var result = document.querySelector('#reconcile-result');
    var commit = document.querySelector('#commit-button');
    var file = function () { return form.querySelector('input[type=file]').files[0]; };
    var formData = function () { var data = new FormData(); if (file()) data.append('file', file()); else form.querySelectorAll('[name]').forEach(function (field) { if (field.name !== 'file') data.append(field.name, field.value); }); return data; };
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var data = formData();
      fetch('/api/reconcile/preview', { method: 'POST', headers: { 'X-CSRF-Token': window.quantCsrf }, body: data })
        .then(function (response) { return response.json().then(function (body) { return { ok: response.ok, body: body }; }); })
        .then(function (response) { if (!response.ok) throw new Error(response.body.detail); result.textContent = JSON.stringify(response.body, null, 2); commit.disabled = false; })
        .catch(function (error) { result.textContent = error.message; });
    });
    commit.addEventListener('click', function () {
      var data = formData();
      fetch('/api/reconcile/commit', { method: 'POST', headers: { 'X-CSRF-Token': window.quantCsrf }, body: data })
        .then(function (response) { return response.json().then(function (body) { return { ok: response.ok, body: body }; }); })
        .then(function (response) { if (!response.ok) throw new Error(response.body.detail); result.textContent = JSON.stringify(response.body, null, 2); commit.disabled = true; })
        .catch(function (error) { result.textContent = error.message; });
    });
  });
}());
